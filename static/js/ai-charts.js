/**
 * AI Charts - Data Visualization Module
 * Uses Chart.js to create interactive visualizations for financial data
 */

// Chart.js defaults for consistent look across visualizations
Chart.defaults.font.family = "'Roboto', 'Segoe UI', sans-serif";
Chart.defaults.animation.duration = 800;
Chart.defaults.plugins.tooltip.cornerRadius = 4;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.elements.bar.borderRadius = 4;
Chart.defaults.elements.line.tension = 0.3;

// Chart instances cache
const chartInstances = {
  featureImportance: null,
  predictionHistory: null,
  portfolioAllocation: null,
};

// Color palettes
const colorPalettes = {
  light: [
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(231, 233, 237, 0.8)',
    'rgba(99, 255, 132, 0.8)',
    'rgba(162, 235, 54, 0.8)',
    'rgba(235, 162, 54, 0.8)',
  ],
  dark: [
    'rgba(64, 185, 254, 0.8)',
    'rgba(255, 129, 152, 0.8)',
    'rgba(255, 216, 106, 0.8)',
    'rgba(105, 222, 222, 0.8)',
    'rgba(173, 142, 255, 0.8)',
    'rgba(255, 179, 84, 0.8)',
    'rgba(200, 200, 200, 0.8)',
    'rgba(129, 255, 152, 0.8)',
    'rgba(192, 245, 84, 0.8)',
    'rgba(255, 182, 74, 0.8)',
  ]
};

/**
 * Update chart theme based on dark/light mode
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function updateChartsTheme(isDarkMode) {
  const theme = isDarkMode ? 'dark' : 'light';
  const textColor = isDarkMode ? '#e0e0e0' : '#333333';
  const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

  // Update Chart.js defaults
  Chart.defaults.color = textColor;
  Chart.defaults.scale.grid.color = gridColor;
  
  // Update existing chart instances
  for (const chartKey in chartInstances) {
    const chart = chartInstances[chartKey];
    if (chart) {
      // Apply theme to each dataset
      chart.data.datasets.forEach((dataset, index) => {
        if (dataset.backgroundColor && Array.isArray(dataset.backgroundColor)) {
          // For datasets with multiple colors (like pie charts)
          dataset.backgroundColor = colorPalettes[theme];
          dataset.borderColor = isDarkMode ? 'rgba(30, 30, 30, 0.8)' : 'rgba(255, 255, 255, 0.8)';
        } else {
          // For datasets with a single color
          dataset.backgroundColor = colorPalettes[theme][index % colorPalettes[theme].length];
          
          // For line charts, use solid colors for the line
          if (dataset.type === 'line') {
            dataset.borderColor = dataset.backgroundColor.replace(', 0.8)', ', 1)');
          }
        }
      });
      
      // Update chart options
      if (chart.options.scales) {
        // Update X and Y axis
        if (chart.options.scales.x) {
          chart.options.scales.x.grid.color = gridColor;
          chart.options.scales.x.ticks.color = textColor;
        }
        if (chart.options.scales.y) {
          chart.options.scales.y.grid.color = gridColor;
          chart.options.scales.y.ticks.color = textColor;
        }
      }
      
      // Update the chart
      chart.update();
    }
  }
}

/**
 * Create a feature importance chart
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Feature importance data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createFeatureImportanceChart(container, data, isDarkMode) {
  if (!container) return;
  
  const theme = isDarkMode ? 'dark' : 'light';
  const features = data.map(item => item.name);
  const values = data.map(item => item.value);
  
  // Clear previous chart if it exists
  if (chartInstances.featureImportance) {
    chartInstances.featureImportance.destroy();
  }
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.featureImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'Feature Importance',
        data: values,
        backgroundColor: colorPalettes[theme],
        borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              return `Importance: ${(context.raw * 100).toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: Math.max(...values) * 1.1,
          ticks: {
            callback: (value) => {
              return `${(value * 100).toFixed(0)}%`;
            }
          }
        },
        y: {
          ticks: {
            padding: 5
          }
        }
      }
    }
  });
}

/**
 * Create a prediction history chart
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Prediction history data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createPredictionHistoryChart(container, data, isDarkMode) {
  if (!container) return;
  
  const theme = isDarkMode ? 'dark' : 'light';
  
  // Clear previous chart if it exists
  if (chartInstances.predictionHistory) {
    chartInstances.predictionHistory.destroy();
  }
  
  // Define colors
  const colors = {
    actual: isDarkMode ? 'rgba(64, 185, 254, 0.8)' : 'rgba(54, 162, 235, 0.8)',
    predicted: isDarkMode ? 'rgba(255, 129, 152, 0.8)' : 'rgba(255, 99, 132, 0.8)',
    correctBg: isDarkMode ? 'rgba(105, 222, 222, 0.2)' : 'rgba(75, 192, 192, 0.2)',
    incorrectBg: isDarkMode ? 'rgba(255, 179, 84, 0.2)' : 'rgba(255, 159, 64, 0.2)'
  };
  
  // Calculate background colors based on prediction accuracy
  const backgroundColors = data.actual.map((actual, index) => {
    const predicted = data.predicted[index];
    return actual === predicted ? colors.correctBg : colors.incorrectBg;
  });
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.predictionHistory = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.dates,
      datasets: [
        {
          type: 'bar',
          label: 'Prediction Background',
          data: data.actual.map(() => 1),
          backgroundColor: backgroundColors,
          barPercentage: 1,
          categoryPercentage: 1,
          order: 3
        },
        {
          type: 'line',
          label: 'Actual',
          data: data.actual.map(val => val === 1 ? 1 : 0),
          borderColor: colors.actual,
          backgroundColor: 'transparent',
          pointBackgroundColor: colors.actual,
          pointRadius: 5,
          order: 1
        },
        {
          type: 'line',
          label: 'Predicted',
          data: data.predicted.map(val => val === 1 ? 1 : 0),
          borderColor: colors.predicted,
          backgroundColor: 'transparent',
          pointBackgroundColor: colors.predicted,
          pointStyle: 'triangle',
          pointRadius: 5,
          order: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            filter: (item) => item.text !== 'Prediction Background'
          }
        },
        tooltip: {
          callbacks: {
            title: (tooltipItems) => {
              return `Date: ${tooltipItems[0].label}`;
            },
            label: (context) => {
              if (context.datasetIndex === 0) return null;
              
              const value = context.raw === 1 ? 'Up' : 'Down';
              return `${context.dataset.label}: ${value}`;
            },
            footer: (tooltipItems) => {
              const index = tooltipItems[0].dataIndex;
              const actual = data.actual[index];
              const predicted = data.predicted[index];
              
              if (actual === predicted) {
                return 'Prediction: Correct';
              } else {
                return 'Prediction: Incorrect';
              }
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          }
        },
        y: {
          type: 'linear',
          min: -0.1,
          max: 1.1,
          ticks: {
            callback: (value) => {
              if (value === 0) return 'Down';
              if (value === 1) return 'Up';
              return '';
            }
          }
        }
      }
    }
  });
}

/**
 * Create a portfolio allocation pie chart
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Portfolio allocation data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createPortfolioAllocationChart(container, data, isDarkMode) {
  if (!container) return;
  
  const theme = isDarkMode ? 'dark' : 'light';
  const symbols = Object.keys(data.weights);
  const weights = Object.values(data.weights);
  
  // Clear previous chart if it exists
  if (chartInstances.portfolioAllocation) {
    chartInstances.portfolioAllocation.destroy();
  }
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.portfolioAllocation = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: symbols,
      datasets: [{
        data: weights,
        backgroundColor: colorPalettes[theme].slice(0, symbols.length),
        borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right'
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.raw;
              const percentage = (value * 100).toFixed(1);
              return `${context.label}: ${percentage}%`;
            }
          }
        }
      }
    }
  });
}

/**
 * Create a correlation matrix heatmap
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Correlation matrix data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createCorrelationHeatmap(container, data, isDarkMode) {
  if (!container) return;
  
  const symbols = Object.keys(data);
  const correlationMatrix = [];
  
  // Build correlation matrix
  symbols.forEach(symbol1 => {
    const row = [];
    symbols.forEach(symbol2 => {
      row.push(data[symbol1][symbol2]);
    });
    correlationMatrix.push(row);
  });
  
  // Define colors for heatmap
  const getColor = (value) => {
    // Generate a color gradient from red (negative) to white (neutral) to blue (positive)
    if (value < 0) {
      // Negative correlation (red)
      const intensity = Math.min(255, Math.floor(-value * 255));
      return isDarkMode 
        ? `rgba(${intensity}, 40, 40, 0.8)`
        : `rgba(${intensity}, 60, 60, 0.8)`;
    } else {
      // Positive correlation (blue)
      const intensity = Math.min(255, Math.floor(value * 255));
      return isDarkMode 
        ? `rgba(40, 40, ${intensity}, 0.8)`
        : `rgba(60, 60, ${intensity}, 0.8)`;
    }
  };
  
  // Set up the chart data
  const datasets = correlationMatrix.map((row, i) => {
    return {
      label: symbols[i],
      data: row.map((value, j) => ({
        x: symbols[j],
        y: symbols[i],
        v: value
      })),
      backgroundColor: row.map(value => getColor(value)),
      borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
      borderWidth: 1
    };
  });
  
  // Clear previous chart if it exists
  if (chartInstances.correlationMatrix) {
    chartInstances.correlationMatrix.destroy();
  }
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.correlationMatrix = new Chart(ctx, {
    type: 'matrix',
    data: {
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            title: () => null,
            label: (context) => {
              const value = context.raw.v;
              return `${context.raw.x} to ${context.raw.y}: ${value.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'category',
          labels: symbols,
          ticks: {
            display: true
          },
          grid: {
            display: false
          }
        },
        y: {
          type: 'category',
          labels: symbols,
          ticks: {
            display: true
          },
          grid: {
            display: false
          }
        }
      }
    }
  });
}

/**
 * Create a price history chart with predictions
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Price history and prediction data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createPricePredictionChart(container, data, isDarkMode) {
  if (!container) return;
  
  const theme = isDarkMode ? 'dark' : 'light';
  
  // Set up datasets
  const historicalDataset = {
    label: 'Historical Price',
    data: data.historical.map((price, i) => ({
      x: data.dates[i],
      y: price
    })),
    borderColor: isDarkMode ? 'rgba(64, 185, 254, 1)' : 'rgba(54, 162, 235, 1)',
    backgroundColor: 'transparent',
    pointRadius: 0,
    borderWidth: 2,
    tension: 0.3
  };
  
  const predictionDataset = {
    label: 'Prediction',
    data: data.historical.slice(-1).concat(data.predictions).map((price, i) => ({
      x: data.dates[data.historical.length - 1 + i],
      y: price
    })),
    borderColor: isDarkMode ? 'rgba(255, 129, 152, 1)' : 'rgba(255, 99, 132, 1)',
    backgroundColor: isDarkMode ? 'rgba(255, 129, 152, 0.2)' : 'rgba(255, 99, 132, 0.2)',
    pointRadius: [5].concat(Array(data.predictions.length).fill(3)),
    pointStyle: 'circle',
    borderWidth: 2,
    borderDash: [5, 5],
    fill: true
  };
  
  // Clear previous chart if it exists
  if (chartInstances.pricePrediction) {
    chartInstances.pricePrediction.destroy();
  }
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.pricePrediction = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [historicalDataset, predictionDataset]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          align: 'end'
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              return `${context.dataset.label}: $${context.raw.y.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'day'
          },
          grid: {
            display: false
          }
        },
        y: {
          title: {
            display: true,
            text: 'Price ($)'
          },
          beginAtZero: false
        }
      }
    }
  });
}

/**
 * Create a returns prediction chart
 * @param {HTMLElement} container - Chart container element
 * @param {Object} data - Returns prediction data
 * @param {boolean} isDarkMode - Whether dark mode is enabled
 */
function createReturnsPredictionChart(container, data, isDarkMode) {
  if (!container) return;
  
  const theme = isDarkMode ? 'dark' : 'light';
  
  // Set up datasets
  const actualDataset = {
    label: 'Actual Returns',
    data: data.actual,
    backgroundColor: isDarkMode ? 'rgba(64, 185, 254, 0.8)' : 'rgba(54, 162, 235, 0.8)',
    borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
    borderWidth: 1,
    categoryPercentage: 0.4,
    barPercentage: 0.8
  };
  
  const predictedDataset = {
    label: 'Predicted Returns',
    data: data.predicted,
    backgroundColor: isDarkMode ? 'rgba(255, 129, 152, 0.8)' : 'rgba(255, 99, 132, 0.8)',
    borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
    borderWidth: 1,
    categoryPercentage: 0.4,
    barPercentage: 0.8
  };
  
  // Clear previous chart if it exists
  if (chartInstances.returnsPrediction) {
    chartInstances.returnsPrediction.destroy();
  }
  
  // Create chart
  const ctx = container.getContext('2d');
  chartInstances.returnsPrediction = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.dates,
      datasets: [actualDataset, predictedDataset]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          align: 'end'
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              return `${context.dataset.label}: ${(context.raw * 100).toFixed(2)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          }
        },
        y: {
          title: {
            display: true,
            text: 'Return (%)'
          },
          ticks: {
            callback: (value) => {
              return `${(value * 100).toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
}

// Export function names to be used in the main JS file
window.aiCharts = {
  updateChartsTheme,
  createFeatureImportanceChart,
  createPredictionHistoryChart,
  createPortfolioAllocationChart,
  createCorrelationHeatmap,
  createPricePredictionChart,
  createReturnsPredictionChart
}; 