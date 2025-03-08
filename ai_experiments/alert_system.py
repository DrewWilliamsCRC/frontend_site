#!/usr/bin/env python3
"""
Alert System Module

This module provides real-time alerting functionality for market predictions
and portfolio changes based on configurable rules and notification methods.
"""

import os
import json
import logging
import smtplib
import time
import threading
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd # type: ignore
import numpy as np # type: ignore
import requests
from apscheduler.schedulers.background import BackgroundScheduler # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('alert_system')

# Load environment variables for notification services
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM')

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')


class AlertRule:
    """
    Base class for alert rules.
    
    Alert rules define conditions that trigger notifications.
    """
    
    def __init__(self, name: str, description: str, enabled: bool = True):
        """
        Initialize alert rule.
        
        Args:
            name (str): Name of the alert rule
            description (str): Description of the alert rule
            enabled (bool): Whether the rule is enabled
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.last_triggered = None
        self.cooldown_hours = 24  # Default cooldown period to avoid alert fatigue
    
    def check(self, context: Dict[str, Any]) -> bool:
        """
        Check if alert should be triggered.
        
        Args:
            context (Dict[str, Any]): Context data for evaluating the rule
            
        Returns:
            bool: True if alert should be triggered, False otherwise
        """
        # Base implementation always returns False
        # Subclasses should override this method
        return False
    
    def get_message(self, context: Dict[str, Any]) -> str:
        """
        Get alert message.
        
        Args:
            context (Dict[str, Any]): Context data for generating the message
            
        Returns:
            str: Alert message
        """
        return f"Alert: {self.name} - {self.description}"
    
    def can_trigger(self) -> bool:
        """
        Check if alert can trigger based on cooldown period.
        
        Returns:
            bool: True if alert can trigger, False if in cooldown
        """
        if self.last_triggered is None:
            return True
            
        cooldown_delta = timedelta(hours=self.cooldown_hours)
        return datetime.now() - self.last_triggered > cooldown_delta
    
    def mark_triggered(self):
        """Mark alert as triggered for cooldown purposes."""
        self.last_triggered = datetime.now()


class PriceAlertRule(AlertRule):
    """
    Price-based alert rule.
    
    Triggers when a symbol's price crosses a threshold.
    """
    
    def __init__(self, name: str, symbol: str, threshold: float, 
                condition: str = 'above', enabled: bool = True):
        """
        Initialize price alert rule.
        
        Args:
            name (str): Name of the alert rule
            symbol (str): Symbol to monitor
            threshold (float): Price threshold
            condition (str): Condition type ('above', 'below', 'cross_above', 'cross_below')
            enabled (bool): Whether the rule is enabled
        """
        description = f"Alert when {symbol} price is {condition} {threshold}"
        super().__init__(name, description, enabled)
        
        self.symbol = symbol
        self.threshold = threshold
        self.condition = condition
        self.last_value = None
        
    def check(self, context: Dict[str, Any]) -> bool:
        """
        Check if price alert should be triggered.
        
        Args:
            context (Dict[str, Any]): Context with current prices
            
        Returns:
            bool: True if alert should be triggered, False otherwise
        """
        if not self.enabled or not self.can_trigger():
            return False
            
        if 'prices' not in context or self.symbol not in context['prices']:
            logger.warning(f"Symbol {self.symbol} not found in context prices")
            return False
            
        current_price = context['prices'][self.symbol]
        triggered = False
        
        if self.condition == 'above':
            triggered = current_price > self.threshold
        elif self.condition == 'below':
            triggered = current_price < self.threshold
        elif self.condition == 'cross_above':
            triggered = (self.last_value is not None and 
                        self.last_value <= self.threshold and 
                        current_price > self.threshold)
        elif self.condition == 'cross_below':
            triggered = (self.last_value is not None and 
                        self.last_value >= self.threshold and 
                        current_price < self.threshold)
        
        # Store current value for next check
        self.last_value = current_price
        
        if triggered:
            self.mark_triggered()
            
        return triggered
    
    def get_message(self, context: Dict[str, Any]) -> str:
        """
        Get price alert message.
        
        Args:
            context (Dict[str, Any]): Context with current prices
            
        Returns:
            str: Alert message
        """
        if 'prices' not in context or self.symbol not in context['prices']:
            return super().get_message(context)
            
        current_price = context['prices'][self.symbol]
        
        if self.condition == 'above':
            return f"🚨 Price Alert: {self.symbol} is trading at ${current_price:.2f}, above your threshold of ${self.threshold:.2f}"
        elif self.condition == 'below':
            return f"🚨 Price Alert: {self.symbol} is trading at ${current_price:.2f}, below your threshold of ${self.threshold:.2f}"
        elif self.condition == 'cross_above':
            return f"🚨 Price Alert: {self.symbol} has crossed above ${self.threshold:.2f} and is now trading at ${current_price:.2f}"
        elif self.condition == 'cross_below':
            return f"🚨 Price Alert: {self.symbol} has crossed below ${self.threshold:.2f} and is now trading at ${current_price:.2f}"
        
        return super().get_message(context)


class PredictionAlertRule(AlertRule):
    """
    ML prediction-based alert rule.
    
    Triggers when a prediction metric crosses a threshold.
    """
    
    def __init__(self, name: str, symbol: str, metric: str, threshold: float,
                condition: str = 'above', enabled: bool = True):
        """
        Initialize prediction alert rule.
        
        Args:
            name (str): Name of the alert rule
            symbol (str): Symbol to monitor
            metric (str): Prediction metric to monitor (e.g., 'confidence', 'probability')
            threshold (float): Threshold value
            condition (str): Condition type ('above', 'below')
            enabled (bool): Whether the rule is enabled
        """
        description = f"Alert when {symbol} {metric} is {condition} {threshold}"
        super().__init__(name, description, enabled)
        
        self.symbol = symbol
        self.metric = metric
        self.threshold = threshold
        self.condition = condition
        
    def check(self, context: Dict[str, Any]) -> bool:
        """
        Check if prediction alert should be triggered.
        
        Args:
            context (Dict[str, Any]): Context with current predictions
            
        Returns:
            bool: True if alert should be triggered, False otherwise
        """
        if not self.enabled or not self.can_trigger():
            return False
            
        if 'predictions' not in context or self.symbol not in context['predictions'] or \
           self.metric not in context['predictions'][self.symbol]:
            logger.warning(f"Symbol {self.symbol} or metric {self.metric} not found in context predictions")
            return False
            
        value = context['predictions'][self.symbol][self.metric]
        
        if self.condition == 'above':
            triggered = value > self.threshold
        elif self.condition == 'below':
            triggered = value < self.threshold
        else:
            triggered = False
        
        if triggered:
            self.mark_triggered()
            
        return triggered
    
    def get_message(self, context: Dict[str, Any]) -> str:
        """
        Get prediction alert message.
        
        Args:
            context (Dict[str, Any]): Context with current predictions
            
        Returns:
            str: Alert message
        """
        if 'predictions' not in context or self.symbol not in context['predictions'] or \
           self.metric not in context['predictions'][self.symbol]:
            return super().get_message(context)
            
        value = context['predictions'][self.symbol][self.metric]
        direction = context['predictions'][self.symbol].get('direction', 'uncertain')
        
        if direction == 'up':
            direction_emoji = "📈"
        elif direction == 'down':
            direction_emoji = "📉"
        else:
            direction_emoji = "📊"
        
        return f"{direction_emoji} Prediction Alert: {self.symbol} {self.metric} is at {value:.2f}, {self.condition} threshold of {self.threshold:.2f}. Predicted direction: {direction}"


class PortfolioAlertRule(AlertRule):
    """
    Portfolio-based alert rule.
    
    Triggers when portfolio metrics change significantly.
    """
    
    def __init__(self, name: str, portfolio_id: str, metric: str, threshold: float,
                condition: str = 'change', enabled: bool = True):
        """
        Initialize portfolio alert rule.
        
        Args:
            name (str): Name of the alert rule
            portfolio_id (str): Portfolio identifier
            metric (str): Portfolio metric to monitor (e.g., 'value', 'return', 'risk')
            threshold (float): Threshold value
            condition (str): Condition type ('change', 'above', 'below')
            enabled (bool): Whether the rule is enabled
        """
        description = f"Alert when portfolio {portfolio_id} {metric} {condition} {threshold}"
        super().__init__(name, description, enabled)
        
        self.portfolio_id = portfolio_id
        self.metric = metric
        self.threshold = threshold
        self.condition = condition
        self.last_value = None
        
    def check(self, context: Dict[str, Any]) -> bool:
        """
        Check if portfolio alert should be triggered.
        
        Args:
            context (Dict[str, Any]): Context with current portfolio metrics
            
        Returns:
            bool: True if alert should be triggered, False otherwise
        """
        if not self.enabled or not self.can_trigger():
            return False
            
        if 'portfolios' not in context or self.portfolio_id not in context['portfolios'] or \
           self.metric not in context['portfolios'][self.portfolio_id]:
            logger.warning(f"Portfolio {self.portfolio_id} or metric {self.metric} not found in context")
            return False
            
        current_value = context['portfolios'][self.portfolio_id][self.metric]
        triggered = False
        
        if self.condition == 'change' and self.last_value is not None:
            # Calculate percentage change
            pct_change = abs((current_value - self.last_value) / self.last_value)
            triggered = pct_change > self.threshold
        elif self.condition == 'above':
            triggered = current_value > self.threshold
        elif self.condition == 'below':
            triggered = current_value < self.threshold
        
        # Store current value for next check
        self.last_value = current_value
        
        if triggered:
            self.mark_triggered()
            
        return triggered
    
    def get_message(self, context: Dict[str, Any]) -> str:
        """
        Get portfolio alert message.
        
        Args:
            context (Dict[str, Any]): Context with current portfolio metrics
            
        Returns:
            str: Alert message
        """
        if 'portfolios' not in context or self.portfolio_id not in context['portfolios'] or \
           self.metric not in context['portfolios'][self.portfolio_id]:
            return super().get_message(context)
            
        current_value = context['portfolios'][self.portfolio_id][self.metric]
        portfolio_name = context['portfolios'][self.portfolio_id].get('name', self.portfolio_id)
        
        if self.condition == 'change' and self.last_value is not None:
            pct_change = (current_value - self.last_value) / self.last_value
            direction = "increased" if pct_change > 0 else "decreased"
            return (f"📢 Portfolio Alert: {portfolio_name} {self.metric} has {direction} by "
                   f"{abs(pct_change):.2%} to {current_value:.2f}, exceeding your threshold of {self.threshold:.2%}")
        elif self.condition == 'above':
            return f"📢 Portfolio Alert: {portfolio_name} {self.metric} is at {current_value:.2f}, above your threshold of {self.threshold:.2f}"
        elif self.condition == 'below':
            return f"📢 Portfolio Alert: {portfolio_name} {self.metric} is at {current_value:.2f}, below your threshold of {self.threshold:.2f}"
        
        return super().get_message(context)


class NotificationChannel:
    """
    Base class for notification channels.
    
    Notification channels are used to send alerts to users.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize notification channel.
        
        Args:
            name (str): Name of the notification channel
            enabled (bool): Whether the channel is enabled
        """
        self.name = name
        self.enabled = enabled
    
    def send(self, message: str, subject: str = "Market Alert") -> bool:
        """
        Send notification.
        
        Args:
            message (str): Notification message
            subject (str): Notification subject
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        # Base implementation always returns False
        # Subclasses should override this method
        return False


class EmailNotifier(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, name: str, recipient_email: str, enabled: bool = True,
                smtp_host: Optional[str] = None, smtp_port: Optional[int] = None,
                smtp_user: Optional[str] = None, smtp_password: Optional[str] = None,
                from_email: Optional[str] = None):
        """
        Initialize email notifier.
        
        Args:
            name (str): Name of the notification channel
            recipient_email (str): Recipient email address
            enabled (bool): Whether the channel is enabled
            smtp_host (str, optional): SMTP server hostname
            smtp_port (int, optional): SMTP server port
            smtp_user (str, optional): SMTP username
            smtp_password (str, optional): SMTP password
            from_email (str, optional): Sender email address
        """
        super().__init__(name, enabled)
        
        self.recipient_email = recipient_email
        self.smtp_host = smtp_host or EMAIL_HOST
        self.smtp_port = smtp_port or EMAIL_PORT
        self.smtp_user = smtp_user or EMAIL_USER
        self.smtp_password = smtp_password or EMAIL_PASSWORD
        self.from_email = from_email or EMAIL_FROM
        
    def send(self, message: str, subject: str = "Market Alert") -> bool:
        """
        Send email notification.
        
        Args:
            message (str): Notification message
            subject (str): Email subject
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Email notifier {self.name} is disabled")
            return False
            
        if not all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_password, self.from_email]):
            logger.error("Email configuration missing")
            return False
            
        try:
            # Create message
            email_message = MIMEMultipart()
            email_message['From'] = self.from_email
            email_message['To'] = self.recipient_email
            email_message['Subject'] = subject
            
            # Add HTML body
            html = f"""
            <html>
              <head></head>
              <body>
                <p>{message}</p>
                <p>This alert was generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
              </body>
            </html>
            """
            email_message.attach(MIMEText(html, 'html'))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(email_message)
                
            logger.info(f"Email notification sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False


class SlackNotifier(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, name: str, webhook_url: Optional[str] = None, enabled: bool = True):
        """
        Initialize Slack notifier.
        
        Args:
            name (str): Name of the notification channel
            webhook_url (str, optional): Slack webhook URL
            enabled (bool): Whether the channel is enabled
        """
        super().__init__(name, enabled)
        
        self.webhook_url = webhook_url or SLACK_WEBHOOK_URL
        
    def send(self, message: str, subject: str = "Market Alert") -> bool:
        """
        Send Slack notification.
        
        Args:
            message (str): Notification message
            subject (str): Notification subject
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Slack notifier {self.name} is disabled")
            return False
            
        if not self.webhook_url:
            logger.error("Slack webhook URL missing")
            return False
            
        try:
            # Prepare Slack message
            payload = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": subject
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        ]
                    }
                ]
            }
            
            # Send message
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
                return True
            else:
                logger.error(f"Error sending Slack notification: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False


class AlertManager:
    """
    Alert manager for handling alert rules and notifications.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.rules = []
        self.notification_channels = []
        self.scheduler = BackgroundScheduler()
        self.running = False
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.
        
        Args:
            rule (AlertRule): Alert rule to add
        """
        with self._lock:
            self.rules.append(rule)
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove alert rule.
        
        Args:
            rule_name (str): Name of the alert rule to remove
            
        Returns:
            bool: True if rule was removed, False if not found
        """
        with self._lock:
            for i, rule in enumerate(self.rules):
                if rule.name == rule_name:
                    del self.rules[i]
                    logger.info(f"Removed alert rule: {rule_name}")
                    return True
            
            logger.warning(f"Alert rule not found: {rule_name}")
            return False
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """
        Add notification channel.
        
        Args:
            channel (NotificationChannel): Notification channel to add
        """
        with self._lock:
            self.notification_channels.append(channel)
            logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_name: str) -> bool:
        """
        Remove notification channel.
        
        Args:
            channel_name (str): Name of the notification channel to remove
            
        Returns:
            bool: True if channel was removed, False if not found
        """
        with self._lock:
            for i, channel in enumerate(self.notification_channels):
                if channel.name == channel_name:
                    del self.notification_channels[i]
                    logger.info(f"Removed notification channel: {channel_name}")
                    return True
            
            logger.warning(f"Notification channel not found: {channel_name}")
            return False
    
    def check_alerts(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all alerts and send notifications.
        
        Args:
            context (Dict[str, Any]): Context data for evaluating rules
            
        Returns:
            List[Dict[str, Any]]: List of triggered alerts
        """
        triggered_alerts = []
        
        with self._lock:
            # Check all rules
            for rule in self.rules:
                try:
                    if rule.check(context):
                        message = rule.get_message(context)
                        
                        # Record triggered alert
                        alert_info = {
                            'rule_name': rule.name,
                            'message': message,
                            'timestamp': datetime.now().isoformat()
                        }
                        triggered_alerts.append(alert_info)
                        
                        # Send notifications
                        self._send_notifications(message, f"Alert: {rule.name}")
                except Exception as e:
                    logger.error(f"Error checking rule {rule.name}: {str(e)}")
        
        return triggered_alerts
    
    def _send_notifications(self, message: str, subject: str) -> None:
        """
        Send notifications to all channels.
        
        Args:
            message (str): Notification message
            subject (str): Notification subject
        """
        for channel in self.notification_channels:
            try:
                success = channel.send(message, subject)
                if not success:
                    logger.warning(f"Failed to send notification via {channel.name}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel.name}: {str(e)}")
    
    def start_monitoring(self, context_provider: Callable[[], Dict[str, Any]], 
                        interval_minutes: int = 5) -> None:
        """
        Start background monitoring.
        
        Args:
            context_provider (Callable): Function that provides context data
            interval_minutes (int): Monitoring interval in minutes
        """
        if self.running:
            logger.warning("Alert monitoring already running")
            return
        
        if not self.scheduler.running:
            self.scheduler.start()
        
        # Define monitoring job
        def monitoring_job():
            try:
                context = context_provider()
                self.check_alerts(context)
            except Exception as e:
                logger.error(f"Error in monitoring job: {str(e)}")
        
        # Schedule job
        self.scheduler.add_job(
            monitoring_job,
            'interval',
            minutes=interval_minutes,
            id='alert_monitoring',
            replace_existing=True
        )
        
        self.running = True
        logger.info(f"Started alert monitoring (interval: {interval_minutes} minutes)")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self.running:
            logger.warning("Alert monitoring not running")
            return
        
        # Remove monitoring job
        self.scheduler.remove_job('alert_monitoring')
        
        # Shut down scheduler if no other jobs
        if not self.scheduler.get_jobs():
            self.scheduler.shutdown()
        
        self.running = False
        logger.info("Stopped alert monitoring")


# Sample context provider for testing
def sample_context_provider() -> Dict[str, Any]:
    """
    Generate sample context for testing.
    
    Returns:
        Dict[str, Any]: Sample context data
    """
    return {
        'prices': {
            'AAPL': 175.50 + np.random.normal(0, 2),
            'MSFT': 385.25 + np.random.normal(0, 3),
            'AMZN': 160.75 + np.random.normal(0, 2),
            'GOOGL': 140.90 + np.random.normal(0, 1.5)
        },
        'predictions': {
            'AAPL': {
                'direction': 'up' if np.random.random() > 0.5 else 'down',
                'confidence': np.random.uniform(0.6, 0.9),
                'target_price': 180.00 + np.random.normal(0, 5)
            },
            'MSFT': {
                'direction': 'up' if np.random.random() > 0.5 else 'down',
                'confidence': np.random.uniform(0.6, 0.9),
                'target_price': 390.00 + np.random.normal(0, 7)
            }
        },
        'portfolios': {
            'default': {
                'name': 'Default Portfolio',
                'value': 100000 + np.random.normal(0, 1000),
                'return': np.random.uniform(-0.02, 0.03),
                'risk': 0.15 + np.random.normal(0, 0.01)
            }
        },
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Example usage
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Add notification channels
    # Uncomment and configure these for real notifications
    # email_notifier = EmailNotifier("Email", "user@example.com")
    # alert_manager.add_notification_channel(email_notifier)
    
    # slack_notifier = SlackNotifier("Slack")
    # alert_manager.add_notification_channel(slack_notifier)
    
    # Add alert rules
    price_rule = PriceAlertRule("AAPL above 180", "AAPL", 180.0, "above")
    alert_manager.add_rule(price_rule)
    
    prediction_rule = PredictionAlertRule("MSFT high confidence", "MSFT", "confidence", 0.85, "above")
    alert_manager.add_rule(prediction_rule)
    
    portfolio_rule = PortfolioAlertRule("Portfolio value change", "default", "value", 0.05, "change")
    alert_manager.add_rule(portfolio_rule)
    
    # Test manual check
    context = sample_context_provider()
    triggered_alerts = alert_manager.check_alerts(context)
    
    if triggered_alerts:
        print("Triggered alerts:")
        for alert in triggered_alerts:
            print(f"- {alert['rule_name']}: {alert['message']}")
    else:
        print("No alerts triggered")
    
    # Start background monitoring (uncomment for real monitoring)
    # alert_manager.start_monitoring(sample_context_provider, interval_minutes=1)
    
    # Run for a while
    # time.sleep(300)
    
    # Stop monitoring
    # alert_manager.stop_monitoring() 