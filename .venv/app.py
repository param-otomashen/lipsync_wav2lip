#only produce avatar response of lip sync. 
#HERE SADTALKER IMAGE IS THERE IN FINAL OUTPUT BUT ORIGINAL WAV2LIP IMAGE AS PLACEHOLDER
import os 
print("model loading..")
from inference_wrapper import load_wav2lip_model
print("model imported")
print("loading wav2lip_gan.pth model from the hardcoded path...")
model= load_wav2lip_model(r"C:\Users\Jasraj\Downloads\wav2lip\wav2lip\wav2lip\checkpoints\wav2lip_gan.pth") #load the model once at the start of the app.
print("model loaded")
from flask import Flask, request, jsonify, send_from_directory, render_template, session
from run import run_inference
from tts import generate_tts, ask_gemini
from flask_session import Session 
import os
import shutil
import tempfile 
import whisper  
print("models imported ")

app = Flask(__name__, static_folder='static')
app.secret_key = "my_secret_key"
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html') #flask looks inside templates folder by default

@app.route('/ask_text', methods=['POST'])
def ask_text():
    print('receiving user message..')
    user_input = request.json['message']  
    print(f"text input = {user_input}")
    #print('sending gemini reply..')
    #response = ask_gemini(user_input)
    #print('yes')
    #tts_path = generate_tts(response)  # saves to examples/output.wav
    tts_path = generate_tts(user_input)
  
    print(f"Generated TTS path: {tts_path}")

    #runs lip sync of the saved examples/output.wav and examples/lady.png
    run_inference(model=model) #and saves output video to result/result_voice.mp4

    source_audio = os.path.abspath("../wav2lip/examples/audio.wav")
    source_video = os.path.abspath("../wav2lip/results/result_voice.mp4") 

    target_audio = os.path.join('static', 'audio.wav') #move the audio, video to render on front end
    target_video = os.path.join('static', 'result_voice.mp4')

    shutil.copyfile(source_audio, target_audio)
    shutil.copyfile(source_video, target_video)
    
    
    # 💾 Store chat history in session
    '''
    history = session.get('chat_history', [])
    history.append({"user": user_input, "bot": response})
    session['chat_history'] = history
    '''
    return jsonify({ 
      #  'response': response,
        'video': '/static/result_voice.mp4',
        'audio': '/static/audio.wav',
     #   'history': history
    }) #received by script.js 

@app.route('/ask_audio', methods=['POST']) 
def ask_audio():
    print("🔵 /ask_audio HIT")  # Entry point log

    audio_file = request.files['audio'] 
    print("📥 Received audio file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        audio_path = temp_audio.name

    print("📄 Saved audio to:", audio_path)

    # Transcribe
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcription = result["text"]
        print("📝 Transcription:", transcription)
    except Exception as e:
        print("❌ Transcription error:", e)
        return jsonify({'error': 'Transcription failed'}), 500

    # Gemini + TTS + Lip sync
    try:
        response = ask_gemini(transcription)
        print("🤖 Gemini response:", response)

        generate_tts(response)
        print("🔊 TTS generated")
        try:
            run_inference(model=model)
            print("🎥 Wav2Lip finished")
        except Exception as e:
            print("❌ Wav2Lip error:", e)
            return jsonify({'error': 'Wav2Lip processing failed'}), 500
        
    except Exception as e:
        print("❌ Processing error:", e)
        return jsonify({'error': 'Processing failed'}), 500

    # File paths
    source_audio = os.path.abspath("../wav2lip/examples/audio.wav")
    source_video = os.path.abspath("../wav2lip/results/result_voice.mp4")
    target_audio = os.path.join('static', 'audio.wav')
    target_video = os.path.join('static', 'result_voice.mp4')

    shutil.copyfile(source_audio, target_audio)
    shutil.copyfile(source_video, target_video)
    
    # Update chat history
    history = session.get('chat_history', [])
    history.append({"user": transcription, "bot": response})
    session['chat_history'] = history
    

    print("✅ Returning response")
    return jsonify({
        'transcription': transcription,
        'response': response,
        'video': '/static/result_voice.mp4',
        'audio': '/static/audio.wav',
        'history': history
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded = False) 

