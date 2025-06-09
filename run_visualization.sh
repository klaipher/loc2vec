#!/bin/bash
# TensorBoard Visualization Runner
# This script starts TensorBoard to view the embeddings generated in the notebook

echo "🗺️  Loc2Vec TensorBoard Viewer"
echo "=============================="

# Check if TensorBoard is available
if ! command -v tensorboard &> /dev/null; then
    echo "❌ TensorBoard not found. Please install it with:"
    echo "   pip install tensorboard"
    exit 1
fi

# Check if visualization data exists
if [ ! -d "runs/loc2vec_projector" ]; then
    echo "❌ No visualization data found!"
    echo ""
    echo "Please run the notebook first to generate embeddings:"
    echo "1. Open and run loc2vec.ipynb"
    echo "2. Complete the training process"
    echo "3. The visualization will be automatically generated"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Found visualization data in runs/loc2vec_projector"
echo ""
echo "🚀 Starting TensorBoard..."
echo "📊 Open http://localhost:6006 in your browser"
echo "📊 Navigate to the PROJECTOR tab to see your embeddings"
echo ""
echo "Press Ctrl+C to stop TensorBoard when you're done exploring."
echo ""

# Start TensorBoard with automatic port detection
echo "🔧 Finding available port..."
for port in 6006 6007 6008 6009; do
    if ! nc -z localhost $port 2>/dev/null; then
        echo "✅ Using port $port"
        echo "📊 TensorBoard will be available at: http://localhost:$port"
        tensorboard --logdir=runs --host=0.0.0.0 --port=$port
        break
    fi
done 