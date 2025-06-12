import {TransformersOptions, Options} from '../../types/options';
import {pipeline} from "@huggingface/transformers";
import {Speech} from '../../speech';

export class Transformers extends Speech {
  private _transcriber: any;
  private _mediaRecorder?: MediaRecorder;
  private _audioChunks: BlobPart[] = [];
  private _stream?: MediaStream;

  async start(options?: Options & TransformersOptions, isDuringReset?: boolean) {
    this.prepareBeforeStart(options);
    await this.startAsync(options, isDuringReset);
  }

  private async startAsync(options?: Options & TransformersOptions, isDuringReset?: boolean) {
    try {
      if (!options?.model) {
        this.setStateOnError('No model specified for Transformers');
        return;
      }
      this._transcriber = await pipeline(
        "automatic-speech-recognition",
        options.model,
        options.pipelineOptions || {}
      );

      // Request microphone access
      this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.setStateOnStart();

      this._audioChunks = [];
      this._mediaRecorder = new MediaRecorder(this._stream);
      this._mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          this._audioChunks.push(event.data);
        }
      };

      this._mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(this._audioChunks, { type: 'audio/webm' });
          const arrayBuffer = await audioBlob.arrayBuffer();
          // HuggingFace pipeline expects an ArrayBuffer or Blob
          const output = await this._transcriber(arrayBuffer);
          const text = output.text || '';
          this.updateElements('', text, text);
          this.finalise(isDuringReset);
          this.setStateOnStop();
        } catch (err) {
          console.error(err);
          this.setStateOnError('Error during transcription: ' + (err as Error).message);
        }
      };

      this._mediaRecorder.start();
    } catch (err) {
      console.error(err);
      this.setStateOnError('Error initializing Transformers: ' + (err as Error).message);
    }
  }

  stop(isDuringReset?: boolean) {
    if (this._mediaRecorder && this._mediaRecorder.state !== 'inactive') {
      this._mediaRecorder.stop();
    }
    if (this._stream) {
      this._stream.getTracks().forEach(track => track.stop());
      this._stream = undefined;
    }
    this.finalise(isDuringReset);
    this.setStateOnStop();
  }
}
