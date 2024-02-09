// 1. Importing required modules i.e. npm install mic sound-play wav stream openai langchain elevenlabs-node dotenv 
import mic from 'mic';
import sound from 'sound-play'
import { Writer } from 'wav';
import { Writable } from 'stream';
import fs, { createWriteStream } from 'fs';
import fs2 from 'node:fs/promises';
import { OpenAI } from 'openai';
import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage } from "langchain/schema";
import voice from 'elevenlabs-node';
import dotenv from 'dotenv';
import { Ollama } from 'ollama-node';

import { pipeline } from '@xenova/transformers';
import wavefile from 'wavefile';

import { DiffusionPipeline } from '@aislamov/diffusers.js'
import { PNG } from 'pngjs'
import { get } from 'http';


const runLocal = true;

dotenv.config();
// 2. Setup for OpenAI and keyword detection.
const openai = new OpenAI();
const keyword = "ivy";
const ollama = new Ollama();
await ollama.setModel("llama2");

let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
const speaker_embeddings = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';
let synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: false });

// 3. Initial microphone setup.
let micInstance = mic({ rate: '16000', channels: '1', debug: false, exitOnSilence: 6 });
let micInputStream = micInstance.getAudioStream();
let isRecording = false;
let audioChunks = [];
// 4. Initiate recording.
const startRecordingProcess = () => {
    console.log("Starting listening process...");
    micInstance.stop();
    micInputStream.unpipe();
    micInstance = mic({ rate: '16000', channels: '1', debug: false, exitOnSilence: 10 });
    micInputStream = micInstance.getAudioStream();
    audioChunks = [];
    isRecording = true;
    micInputStream.pipe(new Writable({
        write(chunk, _, callback) {
            if (!isRecording) return callback();
            audioChunks.push(chunk);
            callback();
        }
    }));
    micInputStream.on('silence', handleSilence);
    micInstance.start();
};
// 5. Handle silence and detection.
const handleSilence = async () => {
    console.log("Detected silence...");
    if (!isRecording) return;
    isRecording = false;
    micInstance.stop();
    const audioFilename = await saveAudio(audioChunks);
    const message = await transcribeAudio(audioFilename);
    if (message && message.toLowerCase().includes(keyword)) {
        console.log("Keyword detected...");
        const responseText = await getAIResponse(message);
        console.log("AI response: ", responseText);
        const fileName = await convertResponseToAudio(responseText);
        console.log("Playing audio...");
        await sound.play('./audio/' + fileName);
        console.log("Playback finished...");
    }
    startRecordingProcess();
};
// 6. Save audio.
const saveAudio = async audioChunks => {
    return new Promise((resolve, reject) => {
        console.log("Saving audio...");
        const audioBuffer = Buffer.concat(audioChunks);
        const wavWriter = new Writer({ sampleRate: 16000, channels: 1 });
        const filename = `${Date.now()}.wav`;
        const filePath = './audio/' + filename;
        wavWriter.pipe(createWriteStream(filePath));
        wavWriter.on('finish', () => {
            resolve(filename);
        });
        wavWriter.on('error', err => {
            reject(err);
        });
        wavWriter.end(audioBuffer);
    });
};


// 7. Transcribe audio.
const transcribeAudioOpenAI = async filename => {
    console.log("Transcribing audio...");
    const audioFile = fs.createReadStream('./audio/' + filename);
    const transcriptionResponse = await openai.audio.transcriptions.create({
        file: audioFile,
        model: "whisper-1",
    });

    console.log(`Transcription done...${transcriptionResponse.text}`);

    return transcriptionResponse.text;
};

const transcribeAudioLocal = async filename => {
    console.log("Transcribing audio local...");
    let b = []
    const fbuff = await fs2.readFile('./audio/' + filename);
    console.log("File buffer: ", b);
    // Read .wav file and convert it to required format
    let wav = new wavefile.WaveFile(fbuff);
    wav.toBitDepth('32f'); // Pipeline expects input as a Float32Array
    wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000
    let audioData = wav.getSamples();
    if (Array.isArray(audioData)) {
        if (audioData.length > 1) {
            const SCALING_FACTOR = Math.sqrt(2);

            // Merge channels (into first channel to save memory)
            for (let i = 0; i < audioData[0].length; ++i) {
                audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
            }
        }

        // Select first channel
        audioData = audioData[0];
    }

    let start = performance.now();
    let output = await transcriber(audioData);
    let end = performance.now();
    console.log(`Execution duration: ${(end - start) / 1000} seconds`);
    console.log(output);

    console.log(`Transcription done...${output.text}`);

    return output.text;
};

const transcribeAudio = runLocal ? transcribeAudioLocal : transcribeAudioOpenAI;



// 8. Communicate with OpenAI.
const getOpenAIResponse = async message => {
    console.log("Communicating with OpenAI...");
    const chat = new ChatOpenAI();
    const response = await chat.call([
        new SystemMessage("You are a helpful voice assistant with a little bit of an attitude.  You give short direct answers and are not afraid to be a little sassy.  You are not a pushover, but you are not mean either. Enclose any emotion expressions in [] like [laughs]"),
        new HumanMessage(message),
    ]);
    return response.text;
};

const getOllamaAIResponse = async message => {
    console.log("Communicating with ollama...");
    ollama.setSystemPrompt("You are a voice assistant with a little bit of an attitude but upbeat and always helpful and ready with a good answer.  You give short and direct answers.  Do not use markdown formatting in the response and don't include emotion instructions in response like *nods* or *laughs*.")
    const resposne = await ollama.generate(message);
    console.log("AI response: ", resposne.output);

    return resposne.output;
};

const getAIResponse = runLocal ? getOllamaAIResponse : getOpenAIResponse;


// 9. Convert response to audio using Eleven Labs.
const convertResponseToAudioElevenLabs = async textInput => {
    const apiKey = process.env.ELEVEN_LABS_API_KEY;
    const voiceId = "XrExE9yKIg1WjnnlVkGX";
    const fileName = `${Date.now()}.mp3`;
    console.log("Converting response to audio...");

    const elevenlabs = new voice({apiKey, voiceId});
    
    // const audioStream = await voice(apiKey, voiceID, text);
    const audioStream = await elevenlabs.textToSpeechStream({textInput});
    const fileWriteStream = fs.createWriteStream('./audio/' + fileName);
    audioStream.pipe(fileWriteStream);
    return new Promise((resolve, reject) => {
        fileWriteStream.on('finish', () => {
            console.log("Audio conversion done...");
            resolve(fileName);
        });
        audioStream.on('error', reject);
    });
};

const convertResponseToAudioLocal = async textInput => {
    
    const fileName = `result-${Date.now()}.wav`;
    console.log("Converting response to audio...");

    const out = await synthesizer(textInput, { speaker_embeddings });

    const wav = new wavefile.WaveFile();
    wav.fromScratch(1, out.sampling_rate, '32f', out.audio);

    try {
        fs.writeFileSync(`./audio/${fileName}`, wav.toBuffer());
    }
    catch (e) {
        console.error(e);
    }
    
    return fileName;
};

const convertResponseToAudio = runLocal ? convertResponseToAudioLocal : convertResponseToAudioElevenLabs;


const usePrompt = async text => {
    const response = await getAIResponse(text);
    const fileName = await convertResponseToAudio(response);
    console.log("Playing audio...");
    await sound.play('./audio/' + fileName);
    console.log("Playback finished...");
}

const imagePrompt = async (text) => {
    const pipe = await DiffusionPipeline.fromPretrained('aislamov/stable-diffusion-2-1-base-onnx', { revision: 'cpu' })
    console.log('pipe', {pipe})
    
    const images = await pipe.run({
        prompt: "an abstract horse illustration",
        numInferenceSteps: 30,
    })

    console.log('images', {images})
    // const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

    // const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
    // p.data = Buffer.from(data.data)
    // p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
    //     console.log('Image saved as output.png');
    // })
}

const prompt = `give me a short summary of the book "The Great Gatsby" by F. Scott Fitzgerald.`;
usePrompt(prompt);
// 10. Start the application and keep it alive.
// startRecordingProcess();

// imagePrompt();
// 11. Keep the process alive.
process.stdin.resume();