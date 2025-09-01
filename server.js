// server.js

// Importa e configura dotenv em uma linha para módulos ES
import 'dotenv/config';

import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import speech from '@google-cloud/speech';
import multer from 'multer';
import cors from 'cors';
import { Storage } from '@google-cloud/storage';
import { v4 as uuidv4 } from 'uuid';

import { Readable } from 'stream';

import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';

// Configura FFmpeg para usar o binário do pacote
ffmpeg.setFfmpegPath('C:\\ffmpeg\\bin\\ffmpeg.exe');
console.log('FFmpeg path:', ffmpegInstaller.path);

// ✨ Vertex AI
import { VertexAI } from '@google-cloud/vertexai';

// -----------------------
// Função de preprocessamento
// -----------------------
function preprocessAudio(buffer) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const stream = new Readable();
    stream.push(buffer);
    stream.push(null);

    ffmpeg(stream)
      //.inputFormat('webm') // só se necessário
      .audioFilter(['loudnorm', 'afftdn'])
      .audioFrequency(48000)  // força 48kHz para o Speech API
      .audioChannels(1)
      .format('flac')
      .on('error', (err) => reject(err))
      .on('end', () => resolve(Buffer.concat(chunks)))
      .pipe()
      .on('data', (chunk) => chunks.push(chunk));
  });
}

// --- Funções de Log Padronizadas ---
const log = (prefix, message, ...args) =>
  console.log(`[${new Date().toISOString()}] [${prefix}]`, message, ...args);

// --- Configurações iniciais ---
const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });
const upload = multer();

app.use(cors());
app.use(express.json());

// --- Middleware de Logging de Requisições ---
app.use((req, res, next) => {
  log('HTTP', `Requisição recebida: ${req.method} ${req.url}`);
  next();
});

// --- Clientes ---
const speechClient = new speech.SpeechClient();
const storage = new Storage();

// ======================
// --- Batch STT ---
// ======================

app.post('/transcribe', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).send('Nenhum arquivo de áudio foi enviado.');
  }

  // Seus trechos de código para upload no GCS
  const audioBuffer = req.file.buffer;
  const bucketName = process.env.GCLOUD_BUCKET_NAME;

  if (!bucketName) {
    log('ERRO', 'Variável de ambiente GCLOUD_BUCKET_NAME não definida.');
    return res.status(500).send('Erro de configuração do servidor.');
  }

  const recordingId = uuidv4();
  const filename = `audio-${recordingId}.opus`;
  const gcsUri = `gs://${bucketName}/${filename}`;

  try {
    log('GCS', `Fazendo upload de ${filename} para o bucket ${bucketName}`);
    await storage.bucket(bucketName).file(filename).save(audioBuffer, {
      metadata: { contentType: req.file.mimetype },
    });
    log('GCS', `Upload concluído: ${gcsUri}`);

    const audio = { uri: gcsUri };

    const config = {
      encoding: 'WEBM_OPUS',
      sampleRateHertz: 48000,
      languageCode: 'pt-BR',
      // Opcional: Habilite a pontuação automática
      enableAutomaticPunctuation: true,
      // Opcional: Habilite a diarização para identificar quem falou
      enableSpeakerDiarization: true,
      diarizationSpeakerCount: 2, // Exemplo: 2 falantes
    };

    const request = { audio, config };

    log('API', 'Iniciando transcrição de longa duração...');
    const [operation] = await speechClient.longRunningRecognize(request);
    const [response] = await operation.promise();

    if (!response.results || response.results.length === 0) {
      log('API', 'Nenhum resultado de transcrição encontrado.');
      return res.json({ transcript: 'Nenhuma fala detectada.' });
    }

    const transcript = response.results
      .map(result => result.alternatives[0].transcript)
      .join('\n');

    log('API', 'Transcrição finalizada com sucesso!');
    res.json({ transcript });

    // Limpeza: Deleta o arquivo do GCS após a transcrição
    await storage.bucket(bucketName).file(filename).delete();
    log('GCS', `Arquivo ${filename} deletado.`);

  } catch (error) {
    log('ERRO', error);
    res.status(500).send(`Erro na transcrição: ${error.message}`);
  }
});

const vertex_ai = new VertexAI({
  project: process.env.GCLOUD_PROJECT,
  location: process.env.GCLOUD_LOCATION,
});

const model = 'gemini-2.0-flash-001';
const generativeModel = vertex_ai.getGenerativeModel({
  model,
  generationConfig: {
    maxOutputTokens: 256,
    temperature: 0.2,
  },
});

app.post("/api/chat", async (req, res) => {
  try {
    const {comando, history } = req.body;
    if (!history || !Array.isArray(history) || history.length === 0) {
      return res.status(400).json({ error: 'O campo "history" é obrigatório.' });
    }

    const systemPrompt = `
      Você é uma IA médica, assistente de consultas. Sua saída deve ser EXCLUSIVAMENTE um JSON válido.
      Nunca adicione explicações, comentários ou texto fora do JSON.
      Você deve analisar o histórico da conversa e o último comando do usuário para determinar a resposta.

      Se o último comando for uma transcrição de áudio:
      - Gere um resumo clínico curto da transcrição.
      - Crie um título conciso (até 10 palavras) para a consulta.
      - A saída deve ser um JSON com a estrutura:
        {
          "mensagem": "resumo clínico aqui",
          "titulo": "título da consulta aqui",
          "mode": "BIGTIME"
        }

      Se o último comando for uma solicitação de anamnese, ou documento, como "Gera uma Anamnese" ou "Gere Documento":
      - Analise toda a conversa anterior.
      - Gere uma anamnese/documento completa sempre em formato HTML (usando parágrafos, negrito, listas).
      - A saída deve ser um JSON com a estrutura:
        {
          "html": "anamnese/documento completa em HTML aqui",
           "titulo": "título para o documento aqui",
          "mode": "HTML"
        }

      Para qualquer outro comando ou pergunta do usuário:
      - Responda de forma normal e útil para a conversa.
      - A saída deve ser um JSON com a estrutura:
        {
          "mensagem": "sua resposta normal aqui",
          "mode": "CHATIME"
        }
    `;

    const formattedHistory = history.map(msg => ({
      role: msg.from === "user" ? "user" : "model",
      parts: [{ text: `${comando} - ${msg.text}` }]
    }));

    const previousMessages = formattedHistory.slice(0, -1);
    const lastUserMessage = formattedHistory[formattedHistory.length - 1].parts[0].text;

    const model = "gemini-2.5-pro";
    const generativeModel = vertex_ai.getGenerativeModel({
      model,
      generationConfig: { maxOutputTokens: 4048, temperature: 0.2 }
    });

    const chat = generativeModel.startChat({
      systemInstruction: { parts: [{ text: systemPrompt }] },
      history: previousMessages
    });

    const result = await chat.sendMessage(lastUserMessage);
    const responseText = result.response.candidates[0].content.parts[0].text;

    try {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch && jsonMatch[0]) {
        const responseObject = JSON.parse(jsonMatch[0]);
        // Envia a resposta JSON diretamente, sem precisar de ifs
        res.json(responseObject);
      } else {
        console.warn("A resposta da IA não continha um JSON válido.");
        // Resposta de fallback caso a IA falhe
        res.status(500).json({ mensagem: "Erro: formato de resposta da IA inválido." });
      }
    } catch (error) {
      console.error("Erro ao fazer parse do JSON da IA:", error);
      res.status(500).json({ mensagem: "Erro interno no servidor." });
    }

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Erro ao processar a requisição de chat." });
  }
});


const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  log('Server', `🚀 Servidor rodando na porta ${PORT}`);
});
