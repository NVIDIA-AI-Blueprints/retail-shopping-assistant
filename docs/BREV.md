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

![Step 1: Create Launchable](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475207726-422f65f0-35a3-487d-b945-56f710a0af67.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T201856Z&X-Amz-Expires=300&X-Amz-Signature=2fff0f4d5f58f342441dc5517351e540716815894fc0d35bfd637a8a633438b6&X-Amz-SignedHeaders=host)

### Step 2: Configure Code Files and Runtime Environment

Configure how you'll provide your code files and select the runtime environment.

1. **Select Code Files Option**: Choose **"I don't have any code files"**
   - We'll clone the repository directly within the environment

2. **Choose Runtime Environment**: Select **"VM Mode"**
   - Provides a virtual machine with Python pre-installed
   - Offers flexibility to install Docker and other dependencies

3. Click **Next** to continue

![Step 2: Configure Environment](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475208291-2f4f0c93-7daa-49cf-95f5-66951ba6e70d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202021Z&X-Amz-Expires=300&X-Amz-Signature=295abfbbc01dd75b6db2051bc76b6c1107b6b47f1a0f56f004fcef409494b408&X-Amz-SignedHeaders=host)

> **Note**: We select "I don't have any code files" because we'll clone the retail shopping assistant repository directly in the VM during setup.

### Step 3: Skip Script Configuration

This optional step allows you to add initialization scripts that run after environment creation.

1. **Skip Script Upload**: Leave this section blank for this deployment
2. **Manual Setup Approach**: We'll configure the retail shopping assistant manually for better control
3. Click **Next** to continue

![Step 3: Script Configuration](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475208789-a36e5951-8ddc-495c-a31e-0c34e045f472.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202159Z&X-Amz-Expires=300&X-Amz-Signature=1b0348f389a1dad2f678c3d119ffa08a496945c93aea59b70471d56c45335b4f&X-Amz-SignedHeaders=host)

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

![Step 4: Jupyter and Network Configuration](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475209222-96d1e038-a3e9-40fa-addc-2adff429c859.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202300Z&X-Amz-Expires=300&X-Amz-Signature=18c7f03d254e65d9e9ed14aaa327a9e2cdc2ab227a3626b0e055ec09e201a33f&X-Amz-SignedHeaders=host)

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

![Step 5: Compute Resources](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475209660-66cfde49-2c75-4019-8ddd-4f36eb026207.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202414Z&X-Amz-Expires=300&X-Amz-Signature=2c50eb9c3c2fab12981b4c185f9099a4540416dd652255c9864d2f86c0604d22&X-Amz-SignedHeaders=host)

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

![Step 6: Configuration Review](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475210130-20e70390-a558-45d4-91f0-721e9564af1f.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202525Z&X-Amz-Expires=300&X-Amz-Signature=d8c0e7b3d1f471923a895132f28d9820e40df1858c949addb85dabf8cf0a0345&X-Amz-SignedHeaders=host)

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

![Step 7: Create Launchable](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475211557-ba0d6b45-599f-48a1-a435-bca8375f0b5c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T202923Z&X-Amz-Expires=300&X-Amz-Signature=1839894757bde3a0e047a6e7d673c918dc3805fe39829e110181db1dae51478b&X-Amz-SignedHeaders=host)

> **Important**: The configuration becomes a shareable template that others can use to deploy identical environments. Instance provisioning and billing begin immediately after creation.

### Step 8: Access Live Deployment Page

After successful template creation, access your deployment options.

1. **Success Confirmation**: Look for the green checkmark confirming creation
2. **Note Deployment URL**: Save the unique Launchable URL provided
3. **Access Deployment**: Click **"View Live Deploy Page"**

![Step 8: Launchable Success](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475211892-87d52092-a5a8-4de8-9692-cadd78e22396.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T203006Z&X-Amz-Expires=300&X-Amz-Signature=aca36de6409dbb8d7778f36ffa53df33d567bb5e8887073e9e50abff1d0045fe&X-Amz-SignedHeaders=host)

> **Next**: The live deploy page provides options to actually provision and access your instance.

### Step 9: Deploy Your Instance

Initiate the actual deployment of your configured environment.

1. **Locate Deploy Button**: Find the green **"Deploy Launchable"** button
2. **Start Deployment**: Click **"Deploy Launchable"** to begin GPU resource provisioning
3. **Monitor Progress**: Wait for the provisioning process to complete

![Step 9: Deploy Launchable](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475212221-df18b9db-807e-468b-8ae2-e463b6d3e680.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T203048Z&X-Amz-Expires=300&X-Amz-Signature=c6408fb790e0ba28d992e330c605f9b66fad58bc53b8f7566e2b2b78301b3ea8&X-Amz-SignedHeaders=host)

> **Billing Note**: GPU resource provisioning and billing begin at this step. The deployment process takes several minutes.

### Step 10: Navigate to Instance Management

Access the instance management interface to monitor deployment progress.

1. **Deployment Status**: Look for "Launchable is now deploying..." message
2. **Access Management**: Click **"Go to Instance Page"** to view progress and access management options

![Step 10: Go to Instance Page](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475214079-f0c68846-cefc-4591-a0d8-5693674e7e41.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T203423Z&X-Amz-Expires=300&X-Amz-Signature=56b74ab8a2bd899a63e028c6609ca7d7963d7a5f0270eb0d587fec21944c76a2&X-Amz-SignedHeaders=host)

> **Management Features**: The instance page provides logs, connection details, and management options during provisioning.

### Step 11: Access Your Running Instance

Wait for instance completion and access the Jupyter environment.

1. **Wait for Completion**: Deployment typically takes 3-5 minutes
2. **Check Status**: Look for green **"Running"** status indicator
3. **Refresh if Needed**: If **"Open Notebook"** appears disabled, refresh the page
4. **Access Jupyter**: Click **"Open Notebook"** to enter the development environment

![Step 11: Instance Running](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475214404-48451a12-6a0e-4c0f-9a6d-a191c3b4672d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T203526Z&X-Amz-Expires=300&X-Amz-Signature=1191bed84fda1d0b766c956bce085e8cbcbed9f35ec1217b81e743f7e757792b&X-Amz-SignedHeaders=host)

> **Instance Controls**: Available options include "Stop" (pause billing), "Delete" (permanently remove), and "Open Notebook" (access environment).

### Step 12: Clone the Repository

Download the retail shopping assistant source code to your instance.

1. **Open Terminal**: In Jupyter, click **"New"** → **"Terminal"**
2. **Clone Repository**: Execute the following command:
   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant.git
   ```
3. **Verify Download**: Confirm the repository files are downloaded successfully

![Step 12: Clone Repository](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475217248-01d34521-cce7-439b-aebc-f75b86e341b1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T204156Z&X-Amz-Expires=300&X-Amz-Signature=6ea823765a9391d35b024196551a8555f28272b3e7c09685ef75ec088f039444&X-Amz-SignedHeaders=host)

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

![Step 13: Deploy Notebook](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475221164-c73c4ccf-4fff-4812-985e-17f19b38918a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T205204Z&X-Amz-Expires=300&X-Amz-Signature=228f68f9ed1de05081eb9023190f43d80167c37beb816ff398dabcf2089d24d4&X-Amz-SignedHeaders=host)

> **Critical**: Execute each notebook cell sequentially to ensure proper setup. The notebook contains all necessary commands and explanations.

### Step 14: Access the Web Interface

Access the retail shopping assistant through your secure tunnel.

1. **Complete Notebook**: Execute all cells until reaching the "Access the Web UI" section
2. **Return to Brev Console**: Navigate back to your instance management page
3. **Use Secure Tunnel**: Click the **shareable URL for port 3000** (e.g., `https://tunnel-xx.brevlab.com:3000`)
4. **Open Application**: The retail shopping assistant web interface opens in your browser

![Step 14: Access Shareable link](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475221665-ae96b25c-78d9-4e89-a720-270ee31d886b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T205336Z&X-Amz-Expires=300&X-Amz-Signature=5d15d0ecd28c547457e0393829b41525f15fd688af7927c1eb62fcbafe57291c&X-Amz-SignedHeaders=host)

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

![Step 15: System Initialization](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475222365-0ee83aad-75b6-46cd-a309-264d800dd944.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T205539Z&X-Amz-Expires=300&X-Amz-Signature=50fd2077b37d52bf8e51275a9f9dea60c9d4198a70fa7b2c19d3a2a339e9fbc5&X-Amz-SignedHeaders=host)

> **Critical**: Wait for complete initialization before interacting with the assistant. Premature interaction may cause errors or incomplete responses.

---

## Deployment Complete! 

🎉 **Congratulations!** You have successfully deployed the NVIDIA Retail Shopping Assistant on Brev.

### Available Features
- **Conversational AI**: Chat with the intelligent shopping assistant
- **Visual Search**: Upload images to find similar products
- **Smart Cart**: Add and manage items in your shopping cart
- **Multi-Agent System**: Experience the full AI-powered retail assistant

![Retail Shopping Assistant](https://github-production-user-asset-6210df.s3.amazonaws.com/2906855/475222617-64dd19ad-71e4-4079-84a2-5e22126c2be4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250806%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250806T205629Z&X-Amz-Expires=300&X-Amz-Signature=88107a6621d23d4f36090545cd46e0705d4babf739e65c1da570aaf58da68a15&X-Amz-SignedHeaders=host)

## Additional Resources

### Documentation
- [User Guide](USER_GUIDE.md) - Complete feature walkthrough
- [API Documentation](API.md) - Technical API reference
- [Deployment Guide](DEPLOYMENT.md) - Alternative deployment methods

### Support
- [NVIDIA Brev Documentation](https://docs.brev.nvidia.com) - Platform-specific help
- [Project Issues](https://github.com/NVIDIA-AI-Blueprints/retail-shopping-assistant/issues) - Report bugs or request features

---

[Back to Documentation Hub](README.md)