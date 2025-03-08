# Contributing to Frontend Site

First off, thank you for considering contributing to this project! 

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible
* Include details about your browser and operating system
* If related to API issues, include any error responses received

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear about it. Just follow these steps:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Include mockups or examples if applicable

### Contributing to AI Dashboard Features

The AI dashboard is a complex feature with several components working together:

1. **Market Data Integration**: When working with market data APIs, ensure proper error handling, rate limiting logic, and caching mechanisms.

2. **Data Visualization**: Chart.js is used for visualizations. Follow the existing patterns for creating and updating charts.

3. **UI Components**: The dashboard uses a card-based layout system. Follow the existing styles and responsive design principles.

4. **Mock Data**: Always implement mock data fallbacks for API-dependent features to ensure the application works when APIs are unavailable.

5. **Error Handling**: Implement robust error handling with appropriate user feedback and graceful degradation.

6. **Performance**: Be mindful of performance, especially with real-time data. Implement lazy loading and optimize DOM operations.

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow our style guides
* End all files with a newline
* Make sure your code passes existing tests
* Add tests for new functionality where applicable
* Update documentation for any changed functionality

## Style Guides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Use conventional commit prefixes (feat:, fix:, docs:, style:, refactor:, perf:, test:, etc.)

### Python Style Guide

* Follow PEP 8 guidelines
* Use 4 spaces for indentation
* Maximum line length of 88 characters (following Black formatter conventions)
* Use docstrings for functions and classes
* Use type hints where appropriate
* Organize imports alphabetically within groups

### JavaScript Style Guide

* Use 2 spaces for indentation
* Use camelCase for variable and function names
* Place spaces around operators
* End statements with semicolons
* 80 character line length
* Prefer `const` over `let`
* Use template literals instead of string concatenation
* Use modern ES6+ features where appropriate

### CSS Style Guide

* Use consistent naming conventions
* Organize properties logically
* Avoid overly specific selectors
* Use responsive design principles
* Implement dark mode for all new styles
* Test across multiple screen sizes

### Documentation Style Guide

* Use [Markdown](https://guides.github.com/features/mastering-markdown/)
* Reference methods and classes in markdown with backticks
* Use language-specific code blocks for examples
* Update documentation when changing functionality
* Include examples for complex features

## Development Environment

### Setting Up Your Development Environment

1. Fork and clone the repository
2. Install dependencies with `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`
3. Set up your environment variables as described in the README
4. Use the provided `dev.sh` script for Docker-based development

### Testing

* Write unit tests for new functionality
* Ensure all tests pass before submitting a pull request
* Test with API mocks to avoid hitting real APIs during testing
* Verify dark mode compatibility for UI changes
* Test on multiple browser sizes for responsive designs

### Debugging

The AI dashboard includes debugging tools:
* Use the debug panel to check component loading status
* Look for console errors in the browser dev tools
* Check the Flask logs for backend errors
* Use mocked data for local development to avoid API rate limits

## Additional Resources

* [Flask Documentation](https://flask.palletsprojects.com/)
* [Chart.js Documentation](https://www.chartjs.org/docs/latest/)
* [Bootstrap Documentation](https://getbootstrap.com/docs/)
* [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/)
* [Guardian API Documentation](https://open-platform.theguardian.com/documentation/)

Thank you for contributing!
