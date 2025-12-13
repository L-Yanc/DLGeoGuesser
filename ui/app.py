"""
GeoGuesser Game Show UI - Backend
A 90s-style game show interface for the GeoLocation Pipeline
"""

import base64
import io
import json
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image

# Add parent directory to path to import pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dl_geoguesser.vision.pipeline import (
    GeoLocationPipeline,
    PipelineConfig,
    save_visualizations,
)
from src.dl_geoguesser.language.nanochat_model import NanoChatModel

app = Flask(__name__)

# Global pipeline instance (loaded once)
pipeline = None
pipeline_ready = False

# Global chat models (loaded on demand)
chat_models = {
    'self_trained_base': None,
    'self_trained_mid': None,
    'self_trained_sft': None,
}
chat_models_ready = False

# Store conversation history per session (simple in-memory storage)
# In production, you'd use Redis or a database
conversation_history = {}

# Store initial LLM analysis per session
llm_initial_analysis = {}

# Default model weights
ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_WEIGHTS = {
    "clip": ROOT_DIR / "runs" / "clip_vibe" / "clip_vibe_precomputed_run" / "best.pt",
    "yolo": ROOT_DIR / "runs" / "yolo" / "M1Pro_run" / "weights" / "best.pt",
    "dino": ROOT_DIR / "runs" / "dino" / "dino_precomputed" / "best.pt",
}


def initialize_pipeline():
    """Initialize the pipeline on first request."""
    global pipeline, pipeline_ready
    
    if pipeline is None:
        print("🎬 Initializing GeoGuesser Game Show Pipeline...")
        config = PipelineConfig(
            clip_weights=str(DEFAULT_WEIGHTS["clip"]),
            yolo_weights=str(DEFAULT_WEIGHTS["yolo"]),
            dino_weights=str(DEFAULT_WEIGHTS["dino"]),
            device="cpu",  # Use CPU for stability
            yolo_conf_threshold=0.4,
            enable_ocr=True,  # Enable OCR to test display
        )
        pipeline = GeoLocationPipeline(config)
        pipeline.load_models()
        pipeline_ready = True
        print("✅ Pipeline ready!")


def initialize_chat_models(force_reload=False):
    """Initialize chat models on first request."""
    global chat_models, chat_models_ready
    
    if chat_models_ready and not force_reload:
        return
    
    print("💬 Initializing Chat Models...")
    
    # Map model types to checkpoint paths (matching chat.py structure)
    checkpoint_paths = {
        'self_trained_base': ROOT_DIR / "models" / "d12_base_1k" / "base_checkpoints",
        'self_trained_mid': ROOT_DIR / "models" / "d12_base_1k" / "mid_checkpoints" / "d12",
        'self_trained_sft': ROOT_DIR / "models" / "d12_base_1k" / "chatsft_checkpoints" / "d12",
    }
    
    for model_type, checkpoint_path in checkpoint_paths.items():
        try:
            if not checkpoint_path.exists():
                print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
                continue
                
            print(f"  Loading {model_type} model from {checkpoint_path}...")
            chat_models[model_type] = NanoChatModel(
                checkpoint_dir=str(checkpoint_path),
                device="cpu"
            )
            print(f"  ✅ {model_type.replace('_', ' ').title()} model loaded")
        except Exception as e:
            print(f"  ⚠️  Failed to load {model_type} model: {e}")
            import traceback
            traceback.print_exc()
    
    chat_models_ready = True
    print(f"✅ Chat models ready! Loaded: {[k for k, v in chat_models.items() if v is not None]}")


@app.route('/')
def index():
    """Render the game show interface."""
    return render_template('index.html')


@app.route('/debug')
def debug():
    """Debug test page."""
    return open(Path(__file__).parent / 'test_debug.html').read()


@app.route('/api/status')
def status():
    """Check if pipeline is ready."""
    return jsonify({
        'ready': pipeline_ready,
        'message': 'Pipeline ready!' if pipeline_ready else 'Loading models...'
    })


@app.route('/api/test-viz')
def test_viz():
    """Test endpoint to check visualization directory."""
    viz_dir = Path(__file__).parent / 'static' / 'results'
    files = list(viz_dir.glob('*')) if viz_dir.exists() else []
    return jsonify({
        'viz_dir': str(viz_dir),
        'exists': viz_dir.exists(),
        'files': [f.name for f in files if f.is_file()],
        'ocr_enabled': pipeline.config.enable_ocr if pipeline else False,
    })


def generate_initial_analysis(result, model_type='self_trained_sft'):
    """
    Generate initial LLM analysis of the image results.
    
    Args:
        result: PipelineResult from image analysis
        model_type: Which LLM to use for analysis
        
    Returns:
        Dict with LLM's decision and reasoning
    """
    try:
        # Initialize models if needed (force reload to ensure all models loaded)
        initialize_chat_models(force_reload=False)
        
        model = chat_models.get(model_type)
        if model is None:
            print(f"⚠️  Model {model_type} not available for initial analysis")
            return None
        
        # Build analysis prompt - keep it concise for better results
        top_3_countries = ', '.join([f"{name} ({score:.0%})" for name, score in sorted(result.country_scores.items(), key=lambda x: x[1], reverse=True)[:3]])
        
        prompt = f"""Analyze this location data and provide your top 2 guesses:

Data:
- Country predictions: {top_3_countries}
- Scene: {result.top_vibe}
- Objects: {', '.join(list(result.detections.keys())) if result.detections else 'none'}
- Text: {result.extracted_text if result.extracted_text else 'none'}

Respond in this format:
1st: [location] - [reason]
2nd: [location] - [reason]

Analysis:"""

        print(f"🤖 Generating initial analysis with {model_type}...")
        print(f"   Prompt length: {len(prompt)} chars")
        
        response = model.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.6,  # Lower for more focused output
            top_k=40,
            stream=False
        )
        
        print(f"✅ Initial analysis: {response[:100]}...")
        
        return {
            'analysis': response.strip(),
            'model': model_type
        }
        
    except Exception as e:
        print(f"❌ Failed to generate initial analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/chat/status')
def chat_status():
    """Check if chat models are ready."""
    return jsonify({
        'ready': chat_models_ready,
        'models': {
            'self_trained_base': chat_models['self_trained_base'] is not None,
            'self_trained_mid': chat_models['self_trained_mid'] is not None,
            'self_trained_sft': chat_models['self_trained_sft'] is not None,
        }
    })


@app.route('/api/chat/generate', methods=['POST'])
def chat_generate():
    """Generate chat response with conversation context."""
    try:
        # Initialize models if needed
        if not chat_models_ready:
            initialize_chat_models()
        
        data = request.json
        user_message = data.get('message', '')
        model_type = data.get('model', 'self_trained_sft')
        session_id = data.get('session_id', 'default')
        context = data.get('context', {})  # Analysis results context
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.8)
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Handle different model types
        if model_type == 'sota':
            # For state-of-the-art, we could integrate OpenAI/Anthropic API
            # For now, return a placeholder
            return jsonify({
                'error': 'State-of-the-art model not yet integrated',
                'suggestion': 'Try "base" or "finetuned" models'
            }), 501
        
        # Get the selected model
        model = chat_models.get(model_type)
        if model is None:
            available = [k for k, v in chat_models.items() if v is not None]
            return jsonify({
                'error': f'Model "{model_type}" not available',
                'available_models': available
            }), 400
        
        # Get or create conversation history for this session
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        history = conversation_history[session_id]
        
        # Build prompt with context and conversation history
        prompt_parts = []
        
        # Add initial LLM analysis if available
        if session_id in llm_initial_analysis:
            initial = llm_initial_analysis[session_id]
            prompt_parts.append("Initial AI Analysis:")
            prompt_parts.append(initial['analysis'])
            prompt_parts.append("")
        
        # Add analysis context if available
        if context:
            prompt_parts.append("Image Analysis Results:")
            if context.get('country'):
                prompt_parts.append(f"- Location: {context['country']['top']} ({context['country']['confidence']}% confidence)")
            if context.get('vibe'):
                prompt_parts.append(f"- Scene type: {context['vibe']['top']} ({context['vibe']['confidence']}% confidence)")
            if context.get('detections') and context['detections'].get('objects'):
                objects = ', '.join(context['detections']['objects'])
                prompt_parts.append(f"- Objects detected: {objects}")
            if context.get('ocr') and context['ocr'].get('text_raw'):
                prompt_parts.append(f"- Text in image: {context['ocr']['text_raw']}")
            prompt_parts.append("")
        
        # Add recent conversation history (last 3 exchanges)
        recent_history = history[-6:] if len(history) > 6 else history
        if recent_history:
            prompt_parts.append("Previous conversation:")
            for msg in recent_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
            prompt_parts.append("")
        
        # Add current user message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response
        print(f"💬 Generating with {model_type} model...")
        print(f"   User: {user_message[:50]}...")
        
        response_text = model.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            stream=False
        )
        
        # Clean up response (remove any prompt echo)
        response_text = response_text.strip()
        
        # Store in conversation history
        history.append({'role': 'user', 'content': user_message})
        history.append({'role': 'assistant', 'content': response_text})
        
        # Keep history manageable (last 20 messages)
        if len(history) > 20:
            conversation_history[session_id] = history[-20:]
        
        return jsonify({
            'success': True,
            'response': response_text,
            'model': model_type,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error generating chat response: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def chat_clear():
    """Clear conversation history for a session."""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in conversation_history:
            del conversation_history[session_id]
        
        return jsonify({'success': True, 'message': 'Conversation cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze an uploaded image."""
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image')
        selected_model = data.get('model', 'self_trained_sft')  # Get selected LLM model
        session_id = data.get('session_id', 'default')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run pipeline
        print(f"Running pipeline on image of size: {image.size}")
        start_time = time.time()
        result = pipeline.predict(image)
        processing_time = time.time() - start_time
        print(f"Pipeline completed in {processing_time:.2f}s")
        
        # Generate visualizations
        # Ensure we're using the correct path (ui/static/results, not ui/ui/static/results)
        viz_dir = Path(__file__).resolve().parent / 'static' / 'results'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: verify correct path
        if 'ui/ui/' in str(viz_dir):
            print(f"⚠️  WARNING: Incorrect path detected: {viz_dir}")
            viz_dir = ROOT_DIR / 'ui' / 'static' / 'results'
            viz_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Corrected to: {viz_dir}")
        
        timestamp = int(time.time() * 1000)
        
        try:
            print(f"🎨 Generating visualizations...")
            print(f"   Output dir: {viz_dir}")
            print(f"   Dir exists: {viz_dir.exists()}")
            print(f"   Detections: {list(result.detections.keys())}")
            print(f"   Top vibe: {result.top_vibe}")
            print(f"   CLIP model: {pipeline._clip is not None}")
            
            saved_files = save_visualizations(
                image=image,
                detections=result.detections,
                output_dir=viz_dir,
                image_name=f"result_{timestamp}.jpg",
                clip_model=pipeline._clip,
                top_class=result.top_vibe,
                create_grid=True,
            )
            
            print(f"✅ Saved files: {saved_files}")
            
            # Only keep the grid image, delete individual images to save space
            grid_path = saved_files.get('grid')
            if grid_path and Path(grid_path).exists():
                # Delete individual images
                for key in ['detections', 'heatmap']:
                    if key in saved_files:
                        try:
                            Path(saved_files[key]).unlink()
                            print(f"   Deleted {key} image (keeping only grid)")
                        except Exception:
                            pass
                
                # Only return grid URL
                viz_urls = {'grid': f"/static/results/{Path(grid_path).name}"}
                print(f"📍 Grid visualization URL: {viz_urls['grid']}")
            else:
                viz_urls = {}
        except Exception as viz_error:
            print(f"❌ Failed to generate visualizations: {viz_error}")
            import traceback
            traceback.print_exc()
            viz_urls = {}
        
        # Format response
        response = {
            'success': True,
            'processing_time': round(processing_time, 2),
            'vibe': {
                'top': result.top_vibe,
                'confidence': round(result.top_vibe_confidence * 100, 1),
                'top_5': [
                    {'name': name, 'confidence': round(score * 100, 1)}
                    for name, score in sorted(
                        result.vibe_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            },
            'country': {
                'top': result.top_country,
                'confidence': round(result.top_country_confidence * 100, 1),
                'top_5': [
                    {'name': name, 'confidence': round(score * 100, 1)}
                    for name, score in sorted(
                        result.country_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            },
            'detections': {
                'count': result.num_detections,
                'objects': list(result.detections.keys())
            },
            'visualizations': viz_urls
        }
        
        # Add OCR data if available
        if result.extracted_text or result.detected_languages:
            # Split text into sentences/phrases for better display
            text_items = []
            if result.extracted_text:
                # Split by common delimiters but keep meaningful chunks
                text_items = [t.strip() for t in result.extracted_text.split('.') if t.strip()]
                if not text_items:  # If no periods, use the whole text
                    text_items = [result.extracted_text]
            
            response['ocr'] = {
                'text': text_items,  # Send as array for easier display
                'text_raw': result.extracted_text,  # Keep raw text too
                'languages': result.detected_languages,
                'has_text': bool(result.extracted_text)
            }
            print(f"OCR data included: {response['ocr']}")
        else:
            print("No OCR data available")
        
        # Generate initial LLM analysis
        print(f"\n🤖 Generating initial LLM analysis with {selected_model}...")
        llm_analysis = generate_initial_analysis(result, model_type=selected_model)
        
        if llm_analysis:
            response['llm_analysis'] = llm_analysis
            # Store for chat context
            llm_initial_analysis[session_id] = llm_analysis
            print(f"✅ LLM analysis added to response")
        else:
            print(f"⚠️  No LLM analysis generated")
        
        print(f"\nFull response keys: {list(response.keys())}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize pipeline before starting server
    initialize_pipeline()
    
    print("\n" + "=" * 60)
    print("🎮 GEOGUESSER GAME SHOW 🎮")
    print("=" * 60)
    print("\n🌐 Open your browser to: http://localhost:5001")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
