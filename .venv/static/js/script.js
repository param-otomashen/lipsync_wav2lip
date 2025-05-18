let mediaRecorder;
let audioChunks = [];
let stream;

document.addEventListener('DOMContentLoaded', function () {
  const messageContainer = document.getElementById('message-container');
  if (messageContainer) {
    messageContainer.innerHTML = '<div id="typing-indicator" class="typing-indicator">AI is typing</div>';
  }
});

function createMessage(text, isBot = false) {
  const messageContainer = document.getElementById('message-container');
  const messageElement = document.createElement('div');
  messageElement.classList.add('message', isBot ? 'bot-message' : 'user-message');
  messageElement.textContent = text;

  const typingIndicator = document.getElementById('typing-indicator');
  messageContainer.insertBefore(messageElement, typingIndicator);
  void messageElement.offsetWidth;
  messageContainer.scrollTop = messageContainer.scrollHeight;
  return messageElement;
}

function showTypingIndicator() {
  const indicator = document.getElementById('typing-indicator');
  indicator.style.display = 'block';
  document.getElementById('message-container').scrollTop = document.getElementById('message-container').scrollHeight;
}

function hideTypingIndicator() {
  document.getElementById('typing-indicator').style.display = 'none';
}

async function askBot() {
  const userInput = document.getElementById("user-input").value;
  if (!userInput.trim()) return;

  createMessage(userInput, false);
  document.getElementById("user-input").value = "";
  showTypingIndicator(); 

  try {
    const res = await fetch('/ask_text', { //sends user_input to flask
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userInput })
    });

    const data = await res.json();
    hideTypingIndicator();

    const messageContainer = document.getElementById('message-container');
    messageContainer.innerHTML = '<div id="typing-indicator" class="typing-indicator">AI is typing</div>';
   /*
    data.history.forEach(entry => {
      createMessage(entry.user, false);
      createMessage(entry.bot, true);
    });
    */

    //document.getElementById("bot-response").innerText = data.response;

    const video = document.getElementById("avatar-video");
    const audio = document.getElementById("avatar-audio");

    const timestamp = new Date().getTime(); //ensures that each request has a unique audio/video timestamp which is played
    video.src = data.video + '?t=' + timestamp;
    audio.src = data.audio + '?t=' + timestamp;


    video.onloadeddata = () => {
      audio.play();
      video.play();
    };
  } catch (error) {
    hideTypingIndicator();
    createMessage("Sorry, I encountered an error. Please try again.", true);
    console.error("Error:", error);
  }
}

document.getElementById('micBtn').addEventListener('click', async () => {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    await startRecording();
  } else {
    stopRecording();
  }
});

document.getElementById('user-input').addEventListener('keypress', function (event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    askBot();
  }
});

function resetMicUI() {
  const micBtn = document.getElementById('micBtn');
  const micStatus = document.getElementById('micStatus');

  micBtn.innerText = "🎤 Talk";
  micBtn.disabled = false;
  micStatus.style.display = 'none';
  micStatus.innerText = "";
  micStatus.style.color = "black";

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    try {
      mediaRecorder.stop();
    } catch (e) {
      console.warn("Tried to stop inactive recorder");
    }
  }

  mediaRecorder = null;
  audioChunks = [];
}

function setupMediaRecorder() {
  mediaRecorder.ondataavailable = event => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    console.log("onstop triggered");

    const micBtn = document.getElementById('micBtn');
    const micStatus = document.getElementById('micStatus');

    micStatus.innerText = "Processing...";
    micStatus.style.color = "orange";
    micBtn.innerText = "Processing...";
    micBtn.disabled = true;

    showTypingIndicator();

    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice.wav');

    try {
      console.log("Sending audio to Flask...");
      const res = await fetch('/ask_audio', { //sends formData i.e. audioBlob to Flask 
        method: 'POST',
        body: formData
      });

      console.log("Received response from Flask");
      const data = await res.json();

      if (data.transcription) {
        document.getElementById('voice-transcription').textContent = data.transcription;
      }

      const messageContainer = document.getElementById('message-container');
      messageContainer.innerHTML = '';

      data.history.forEach(entry => {
        createMessage(entry.user, false);
        createMessage(entry.bot, true);
      });

      const video = document.getElementById("avatar-video");
      const audio = document.getElementById("avatar-audio");

      video.src = data.video;
      audio.src = data.audio;

      video.onloadeddata = () => {
        try {
          audio.play();
          video.play();
        } catch (err) {
          console.error("Playback error:", err);
        }
      };
    } catch (error) {
      createMessage("Sorry, I encountered an error. Please try again.", true);
      console.error("Fetch or JSON parse error:", error);
    } finally {
      console.log("Resetting mic UI...");
      resetMicUI();
      hideTypingIndicator();
    }
  };
}

async function startRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    console.warn("Recorder active, stopping before starting new one...");
    return new Promise(resolve => {
      mediaRecorder.onstop = async () => {
        console.log("Stopped old recorder. Restarting...");
        resetMicUI();
        resolve(startRecording());
      };
      mediaRecorder.stop();
    });
  }

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  setupMediaRecorder();
  mediaRecorder.start();

  const micBtn = document.getElementById('micBtn');
  const micStatus = document.getElementById('micStatus');

  micBtn.innerText = "⏹ Stop";
  micStatus.innerText = "Recording...";
  micStatus.style.color = "red";
  micStatus.style.display = 'block';
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    console.log("Stopping recording...");
    mediaRecorder.stop();
  }
}
