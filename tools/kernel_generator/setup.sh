#!/bin/bash
#
# Setup script for TT-Metal Kernel Generator
#

set -e

echo "Setting up TT-Metal Kernel Generator..."

# Check if we're in the right directory
if [[ ! -f "generate_kernel.py" ]]; then
    echo "Error: Please run this script from the kernel_generator directory"
    exit 1
fi

# Check for Python 3.8+
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python $python_version found"
else
    echo "✗ Python 3.8+ required, found $python_version"
    exit 1
fi

# Check for TT_METAL_HOME
if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Setting TT_METAL_HOME to parent directory..."
    export TT_METAL_HOME=$(realpath ../../..)
    echo "TT_METAL_HOME=$TT_METAL_HOME"
else
    echo "✓ TT_METAL_HOME=$TT_METAL_HOME"
fi

# Check for OpenAI API key
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠ Warning: OPENAI_API_KEY not set"
    echo "  Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
else
    echo "✓ OpenAI API key is set"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Create output directory
mkdir -p generated_examples
echo "✓ Created output directory: generated_examples"

# Make scripts executable
chmod +x generate_kernel.py
echo "✓ Made generate_kernel.py executable"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage examples:"
echo "  ./generate_kernel.py --operation add --core-mode single"
echo "  ./generate_kernel.py --operation multiply --core-mode multi --refine --generate-host"
echo "  ./generate_kernel.py --operation subtract --core-mode multi --refine --generate-host --validate"
echo ""
echo "For more help: ./generate_kernel.py --help"
