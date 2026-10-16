export function startContinuousListening(onResult, onError) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    onError("Speech recognition isn't supported in this browser. Try Chrome.");
    return null;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = false;

  let shouldKeepListening = true;

  recognition.onresult = (event) => {
    const lastResult = event.results[event.results.length - 1];
    const text = lastResult[0].transcript.toLowerCase().trim();
    onResult(text);
  };

  recognition.onerror = (event) => {
    if (event.error !== "no-speech") {
      onError(`Speech recognition error: ${event.error}`);
    }
  };

  recognition.onend = () => {
    if (shouldKeepListening) {
      recognition.start();
    }
  };

  recognition.start();

  return {
    stop: () => {
      shouldKeepListening = false;
      recognition.stop();
    },
  };
}