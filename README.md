# Installation Instructions

## Overview
This repository contains multiple components, each with its own set of dependencies and installation instructions. Please follow the appropriate guide for the specific component you're interested in.

### 1. GWAE 
To install and set up the GWAE component, follow the instructions in the `README_gwae.md` file located inside the `GWAE` folder.

If you prefer not to follow the detailed installation guide in the `README_gwae.md`, you can instead use the environment provided in the `environment_gwae.yaml` file. This environment contains the dependencies on which the GWAE implementation was tested.

### 2. GWGAN
Similarly, for the GWGAN component, go to the `GWGAN` folder and read the `README_gwgan.md` for detailed installation instructions.

Alternatively, you can use the `environment_gwgan.yaml` file to set up the environment with all the necessary dependencies tested for the GWGAN implementation.

### 3. Power Spherical Package Installation
Before running any of the components, you need to install the `power_spherical` package. You can find it at the following repository:

[Power Spherical GitHub Repository](https://github.com/nicola-decao/power_spherical)

Follow the installation instructions in that repository, and then run:

```bash
pip install -r requirements.txt
