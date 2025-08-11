# Deploying Retail Shopping Assistant on NVIDIA Brev

This comprehensive guide provides step-by-step instructions for deploying the Retail Shopping Assistant on NVIDIA Brev using GPU Environment Templates (Launchables).

## Overview

NVIDIA Brev provides GPU Environment Templates called "Launchables" that enable one-click deployment of GPU-accelerated applications. Launchables include pre-configured compute resources, containers, and secure networking accessible via shareable URLs.

## Prerequisites

Before starting this deployment, ensure you have:

- Access to the NVIDIA Brev platform ([brev.nvidia.com](https://brev.nvidia.com))
- NVIDIA NGC API key ([Get your API key here](https://ngc.nvidia.com/))
- Basic familiarity with containerized applications and Jupyter notebooks

## Step-by-Step Deployment Guide

### Step 1: Access the NVIDIA Brev Platform

1. Navigate to [brev.nvidia.com](https://brev.nvidia.com)
2. Click **Launchables** in the top navigation menu
3. Click **Create Launchable** to begin creating your GPU environment template

![Step 1: Create Launchable](images/step1.png)

### Step 2: Configure Code Files and Runtime Environment

Configure how you'll provide your code files and select the runtime environment.

1. **Select Code Files Option**: Choose **"I don't have any code files"**
   - We'll clone the repository directly within the environment

2. **Choose Runtime Environment**: Select **"VM Mode"**
   - Provides a virtual machine with Python pre-installed
   - Offers flexibility to install Docker and other dependencies

3. Click **Next** to continue

![Step 2: Configure Environment](images/step2.png)

> **Note**: We select "I don't have any code files" because we'll clone the retail shopping assistant repository directly in the VM during setup.

### Step 3: Skip Script Configuration

This optional step allows you to add initialization scripts that run after environment creation.

1. **Skip Script Upload**: Leave this section blank for this deployment
2. Click **Next** to continue

![Step 3: Script Configuration](images/step3.png)

> **Note**: The script upload feature is experimental. Manual setup provides better control over the installation process.

### Step 4: Configure Jupyter and Network Access

Configure your development environment and network access for the retail shopping assistant.

#### Jupyter Notebook Configuration
1. **Enable Jupyter**: Select **"Yes, Install Jupyter on the Host (Recommended)"**
   - Provides a convenient development environment for testing and debugging

#### Network Access Configuration
2. **Configure Secure Tunnel**: Set up secure external access to the application
   - **Secure Link Name**: Use `tunnel-1` (or keep the default)
   - **Port**: Enter `3000` (the retail shopping assistant's default UI port)

3. Click **Next** to continue

![Step 4: Jupyter and Network Configuration](images/step4.png)

> **Important**: Port 3000 is the default port for the retail shopping assistant's React frontend. The secure tunnel provides external access to the application.

### Step 5: Select Compute Resources

Configure the GPU compute resources for optimal performance.

#### Recommended Configuration
1. **Select GPU Type**: Choose **H100** from the available options
2. **Select Configuration**: Choose **4x NVIDIA H100** for optimal performance
   - **Specifications**: 4x H100 GPUs with 80GB VRAM each
   - **Memory**: High-RAM configuration (varies by provider)
   - **Storage**: Flexible storage options

#### Alternative Configurations
If 4x H100 is unavailable or budget is a concern:
- **2x NVIDIA H100**: Reduced performance, suitable for smaller workloads
- **2x NVIDIA A100**: Balanced performance and cost

3. Click **Next** to review your configuration

![Step 5: Compute Resources](images/step1.png)

> **Performance Note**: The retail shopping assistant is optimized for 4x H100 GPUs as specified in the main README. This ensures smooth operation of all AI models including embeddings, LLMs, and NIMs.

### Step 6: Review Configuration Summary

Review your selected configuration and pricing information.

1. **Review Configuration Details**:
   - **Compute**: Selected GPU configuration (e.g., 2x NVIDIA H100)
   - **Storage**: Disk storage allocation (e.g., 5TB SSD)
   - **Network**: Configured tunnels (tunnel-1:3000)
   - **Pricing**: Hourly rate

2. **Verify Settings**: Ensure all configurations meet your requirements
3. Click **Next** to proceed

![Step 6: Configuration Review](images/step6.png)

> **Cost Warning**: Note the hourly rate. Brev instances cannot be stopped/restarted—only deleted. Plan your usage accordingly.

### Step 7: Create Your Launchable Template

Create your GPU environment template with the configured settings.

1. **Final Configuration Review**:
   - **Compute**: GPU configuration (e.g., NVIDIA H100 with 2 GPUs × 52 CPUs)
   - **Container**: VM Mode with Jupyter enabled
   - **Exposed Ports**: tunnel-1:3000 for web access

2. **Name Your Launchable**: Enter a descriptive name:
   - Example: `retail-shopping-assistant`
   - Use your preferred naming convention

3. **Create Template**: Click **"Create Launchable"**

![Step 7: Create Launchable](images/step7.png)

> **Important**: The configuration becomes a shareable template that others can use to deploy identical environments. Instance provisioning and billing begin immediately after creation.

### Step 8: Access Live Deployment Page

After successful template creation, access your deployment options.

1. **Success Confirmation**: Look for the green checkmark confirming creation
2. **Note Deployment URL**: Save the unique Launchable URL provided
3. **Access Deployment**: Click **"View Live Deploy Page"**

![Step 8: Launchable Success](images/step8.png)

> **Next**: The live deploy page provides options to actually provision and access your instance.

### Step 9: Deploy Your Instance

Initiate the actual deployment of your configured environment.

1. **Locate Deploy Button**: Find the green **"Deploy Launchable"** button
2. **Start Deployment**: Click **"Deploy Launchable"** to begin GPU resource provisioning
3. **Monitor Progress**: Wait for the provisioning process to complete

![Step 9: Deploy Launchable](images/step9.png)

> **Billing Note**: GPU resource provisioning and billing begin at this step. The deployment process takes several minutes.

### Step 10: Navigate to Instance Management

Access the instance management interface to monitor deployment progress.

1. **Deployment Status**: Look for "Launchable is now deploying..." message
2. **Access Management**: Click **"Go to Instance Page"** to view progress and access management options

![Step 10: Go to Instance Page](images/step10.png)

> **Management Features**: The instance page provides logs, connection details, and management options during provisioning.

### Step 11: Access Your Running Instance

Wait for instance completion and access the Jupyter environment.

1. **Wait for Completion**: Deployment typically takes 3-5 minutes
2. **Check Status**: Look for green **"Running"** status indicator
3. **Refresh if Needed**: If **"Open Notebook"** appears disabled, refresh the page
4. **Access Jupyter**: Click **"Open Notebook"** to enter the development environment

![Step 11: Instance Running](images/step11.png)

> **Instance Controls**: Available options include "Stop" (pause billing), "Delete" (permanently remove), and "Open Notebook" (access environment).

### Step 12: Clone the Repository

Download the retail shopping assistant source code to your instance.

1. **Open Terminal**: In Jupyter, click **"New"** → **"Terminal"**
2. **Clone Repository**: Execute the following command:
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
   ```
3. **Verify Download**: Confirm the repository files are downloaded successfully

![Step 12: Clone Repository](images/step12.png)

> **Next Step**: With the source code available, proceed to configure and deploy the application.

### Step 13: Follow the Deployment Notebook

Use the included deployment notebook to automate the setup process.

1. **Navigate to Files**: In Jupyter's file browser (left panel), browse to the cloned repository
2. **Open Deployment Notebook**: Click **`1_Deploy_Retail_Shopping_Assistant.ipynb`** in the `/notebook/` directory
3. **Execute All Cells**: Follow the notebook's step-by-step instructions:
   - Obtain your NVIDIA API key from NGC
   - Configure environment variables
   - Start Docker services
   - Verify deployment status

![Step 13: Deploy Notebook](images/step12.png)

> **Critical**: Execute each notebook cell sequentially to ensure proper setup. The notebook contains all necessary commands and explanations.

### Step 14: Access the Web Interface

Access the retail shopping assistant through your secure tunnel.

1. **Complete Notebook**: Execute all cells until reaching the "Access the Web UI" section
2. **Return to Brev Console**: Navigate back to your instance management page
3. **Use Secure Tunnel**: Click the **shareable URL for port 3000** (e.g., `https://tunnel-xx.brevlab.com:3000`)
4. **Open Application**: The retail shopping assistant web interface opens in your browser

![Step 14: Access Shareable link](images/step14.png)

> **Important**: Use the Brev secure tunnel URL, not `http://localhost:3000` mentioned in the notebook.

### Step 15: Wait for System Initialization

Allow the system to complete initialization before use.

1. **Monitor Initialization**: The system automatically creates embeddings for products and images
2. **Check Progress**: Observe initialization in the deployment notebook output or terminal logs
3. **Wait for Completion**: Process typically takes **2-5 minutes** depending on GPU configuration
4. **Watch for Completion Indicators**:
   - "Processing image batch" (image embeddings)
   - "Milvus database ready" (vector database initialization)
   - "Uvicorn running" (web server ready)

![Step 15: System Initialization](images/step15.png)

> **Critical**: Wait for complete initialization before interacting with the assistant. Premature interaction may cause errors or incomplete responses.

---

## Deployment Complete! 

🎉 **Congratulations!** You have successfully deployed the NVIDIA Retail Shopping Assistant on Brev.

### Available Features
- **Conversational AI**: Chat with the intelligent shopping assistant
- **Visual Search**: Upload images to find similar products
- **Smart Cart**: Add and manage items in your shopping cart
- **Multi-Agent System**: Experience the full AI-powered retail assistant

![Retail Shopping Assistant](images/step16.jpg)

## Additional Resources

### Documentation
- [User Guide](USER_GUIDE.md) - Complete feature walkthrough
- [API Documentation](API.md) - Technical API reference
- [Deployment Guide](DEPLOYMENT.md) - Alternative deployment methods

### Support
- [NVIDIA Brev Documentation](https://docs.nvidia.com/brev/latest/index.html) - Platform-specific help
- [Project Issues](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues) - Report bugs or request features

---

[Back to Documentation Hub](README.md)
