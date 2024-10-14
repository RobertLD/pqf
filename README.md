## Project Overview

PQF is designed to provide a comprehensive suite of indicators, research tools, and statistical measures for quantitative finance. Built on the Polars library, enabling fast data manipulation and computation.

### Documentation
https://robertld.github.io/pqf/

## Contributing

Follow these steps to set up your development environment using the provided devcontainer:

> Note: If you use an IDE other than VSCode, setting up the project is left up to the reader.

1. Setting Up the Development Environment

    - **Install VS Code**: Ensure you have Visual Studio Code installed on your machine.
    - **Install the Remote Development Extension**: Install the Remote Development extension pack in VS Code.

2. Using the DevContainer

    - **Clone the Repository**:

    ```bash

    git clone https://github.com/RobertLD/pqf.git
    cd pqf
    ```

    - **Open in VS Code**: Launch VS Code and open the cloned repository. It should prompt you to reopen in the container.

    - **Reopen in Container**: Click on Reopen in Container. VS Code will build the devcontainer as specified in the .devcontainer directory, setting up all necessary dependencies.

3. Managing Dependencies with Poetry

    - **Install Project Dependencies** : Once inside the devcontainer, run the following command to install project dependencies:

    ```bash
    poetry install
    ```

4. Running Tests

    Ensure your changes do not break existing functionality by running the tests:

    ```bash

    poetry run pytest
    ```

5. Submitting your code for review
    - **Commit and Push**: After verifying your changes, commit them and push your branch:

    ```bash

    git add .
    git commit -m "Add my feature"
    git push origin feature/my-feature
    ```
    - **Open a Pull Request**: Go to the repository on GitHub and open a pull request with a description of your changes.