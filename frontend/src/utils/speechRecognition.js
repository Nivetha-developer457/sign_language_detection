export async function startContinuousListening(onResult, onError, onInterim) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    onError("Speech recognition isn't supported in this browser. Try Chrome.");
    return null;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    onError("Microphone access is unavailable. Open this app in Chrome on localhost or HTTPS.");
    return null;
  }

  let mediaStream;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    if (["NotFoundError", "NotReadableError"].includes(error.name)) {
      const audioInputs = await navigator.mediaDevices.enumerateDevices();
      for (const device of audioInputs.filter((item) => item.kind === "audioinput" && item.deviceId)) {
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { deviceId: { exact: device.deviceId } },
          });
          break;
        } catch {
          // Try the next available input device.
        }
      }
    }

    if (!mediaStream) {
      const messages = {
        NotAllowedError: "Microphone access was blocked. Allow microphone access for this site and try again.",
        NotFoundError: "No microphone was detected. Connect a microphone and try again.",
        NotReadableError: "The microphone is busy in another app. Close that app and try again.",
      };
      onError(messages[error.name] || `Could not access microphone: ${error.message}`);
      return null;
    }
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = true;

  let shouldKeepListening = true;

  const startRecognition = () => {
    try {
      recognition.start();
    } catch (error) {
      shouldKeepListening = false;
      mediaStream.getTracks().forEach((track) => track.stop());
      onError(`Could not start microphone: ${error.message}`);
    }
  };

  recognition.onresult = (event) => {
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const text = result[0].transcript.toLowerCase().trim();
      if (!text) continue;

      if (result.isFinal) {
        onResult(text);
      } else {
        onInterim?.(text);
      }
    }
  };

  recognition.onerror = (event) => {
    if (event.error !== "no-speech") {
      shouldKeepListening = false;
      const messages = {
        "not-allowed": "Microphone access was blocked. Allow microphone access for this site and try again.",
        "audio-capture": "No microphone was found. Connect a microphone and try again.",
        network: "Speech recognition needs an internet connection.",
      };
      mediaStream.getTracks().forEach((track) => track.stop());
      onError(messages[event.error] || `Speech recognition error: ${event.error}`);
    }
  };

  recognition.onend = () => {
    if (shouldKeepListening) {
      startRecognition();
    }
  };

  startRecognition();

  return {
    stop: () => {
      shouldKeepListening = false;
      recognition.stop();
      mediaStream.getTracks().forEach((track) => track.stop());
    },
  };
}