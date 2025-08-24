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

// ✨ Vertex AI
import { VertexAI } from '@google-cloud/vertexai';

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
const vertex_ai = new VertexAI({
  project: process.env.GCLOUD_PROJECT,
  location: process.env.GCLOUD_LOCATION,
});

const model = 'gemini-1.5-flash-001'; // Modelo atualizado
const generativeModel = vertex_ai.getGenerativeModel({
  model,
  generationConfig: {
    maxOutputTokens: 8192,
    temperature: 0.2,
  },
});

/**
 * Função auxiliar para chamar o Vertex AI e centralizar o logging.
 */
const callVertexAI = async (endpointName, prompt, generationConfig = {}) => {
  log('VertexAI', `Iniciando chamada para o endpoint: ${endpointName}`);
  log(
    'VertexAI',
    `Prompt enviado:\n---INÍCIO DO PROMPT---\n${prompt}\n---FIM DO PROMPT---`
  );

  const request = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { ...generativeModel.generationConfig, ...generationConfig },
  };

  const result = await generativeModel.generateContent(request);
  const generatedText = result.response.candidates[0].content.parts[0].text;

  log('VertexAI', `Resposta recebida do endpoint ${endpointName}`);
  return generatedText.trim();
};

// ===================================
// --- WebSocket STT com Automação ---
// ===================================
io.on('connection', (socket) => {
  log('WebSocket', `Cliente conectado: ${socket.id}`);

  let recognizeStream = null;
  let recognitionConfig = null;
  let silenceTimer = null;
  let streamRestartTimer = null;
  const silenceTimeoutDuration = 10000; // 10 segundos
  const maxStreamDuration = 290 * 1000; // ~4.8 minutos

  const stopRecognizeStream = () => {
    if (recognizeStream) {
      recognizeStream.end();
      recognizeStream = null;
      log('WebSocket', `Stream de reconhecimento encerrado para: ${socket.id}`);
    }
    clearTimeout(streamRestartTimer);
    clearTimeout(silenceTimer);
  };

  const startRecognizeStream = () => {
    if (recognizeStream || !recognitionConfig) {
      log(
        'WebSocket',
        `Tentativa de iniciar stream falhou (já iniciado ou sem config) para: ${socket.id}`
      );
      return;
    }
    log(
      'WebSocket',
      `Iniciando/Reiniciando stream de reconhecimento para: ${socket.id}`
    );

    const request = { config: recognitionConfig, interimResults: true };

    recognizeStream = speechClient
      .streamingRecognize(request)
      .on('error', (err) => {
        log('SpeechAPI-ERROR', `Erro no streaming para ${socket.id}:`, err.message);
        socket.emit('error', 'Erro no reconhecimento de fala.');
        stopRecognizeStream();
      })
      .on('data', (data) => {
        const result = data.results[0];
        if (result && result.alternatives[0]) {
          socket.emit('transcript-data', {
            text: result.alternatives[0].transcript,
            isFinal: result.isFinal,
            timestamp: new Date().toLocaleTimeString('pt-BR', {
              hour: '2-digit',
              minute: '2-digit',
            }),
            speakerTag:
              result.alternatives[0].words?.[
                result.alternatives[0].words.length - 1
              ]?.speakerTag,
          });
        }
      });

    // reinício automático do stream depois de 4.8 minutos
    streamRestartTimer = setTimeout(() => {
      log(
        'WebSocket',
        `Stream atingiu a duração máxima de ${
          maxStreamDuration / 1000
        }s. Reiniciando para ${socket.id}...`
      );
      stopRecognizeStream();
      startRecognizeStream();
    }, maxStreamDuration);
  };

  const resetSilenceTimer = () => {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(() => {
      log(
        'WebSocket',
        `Silêncio detectado para ${socket.id}. Reiniciando recognizeStream no servidor.`
      );

      // 🟢 reinicia internamente sem pedir nada ao cliente
      stopRecognizeStream();
      startRecognizeStream();
    }, silenceTimeoutDuration);
  };

  socket.on('start-recording', (config) => {
    log('WebSocket', `Evento 'start-recording' recebido de ${socket.id}`, config);
    recognitionConfig = {
      encoding: 'WEBM_OPUS',
      sampleRateHertz: config.sampleRateHertz || 48000,
      languageCode: config.lang || 'pt-BR',
      enableAutomaticPunctuation: true,
      diarizationConfig: {
        enableSpeakerDiarization: true,
        minSpeakerCount: 2,
        maxSpeakerCount: 6,
      },
      model: 'telephony',
      useEnhanced: true,
    };
    stopRecognizeStream();
    startRecognizeStream();
    resetSilenceTimer();
  });

  socket.on('audio-data', (data) => {
    if (recognizeStream && data) {
      recognizeStream.write(data);
      resetSilenceTimer();
    } else if (!recognizeStream) {
      log(
        'WebSocket',
        `Recebido 'audio-data' de ${socket.id}, mas o stream não está pronto. Ignorando chunk.`
      );
    }
  });

  socket.on('force-flush-partial', (partial) => {
    log('WebSocket', `Evento 'force-flush-partial' recebido de ${socket.id}`);
    socket.emit('transcript-data', { ...partial, isFinal: true });
  });

  socket.on('stop-recording', () => {
    log('WebSocket', `Evento 'stop-recording' recebido de ${socket.id}`);
    stopRecognizeStream();
  });

  socket.on('disconnect', () => {
    log('WebSocket', `Cliente desconectado: ${socket.id}`);
    stopRecognizeStream();
  });
});

// ======================
// --- Batch STT ---
// ======================
app.post('/batch-transcribe', upload.single('file'), async (req, res) => {
  const endpointName = '/batch-transcribe';
  log('API', `Iniciando ${endpointName}`);
  try {
    if (!req.file) {
      log('API-ERROR', `${endpointName} - Nenhum arquivo enviado.`);
      return res.status(400).json({ error: 'Nenhum arquivo enviado.' });
    }

    const audioBuffer = req.file.buffer;
    const bucketName = process.env.GCLOUD_BUCKET_NAME;
    const recordingId = uuidv4();
    const filename = `audio-${recordingId}.opus`;
    const gcsUri = `gs://${bucketName}/${filename}`;

    log('GCS', `Fazendo upload de ${filename} para o bucket ${bucketName}`);
    await storage.bucket(bucketName).file(filename).save(audioBuffer, {
      metadata: { contentType: req.file.mimetype },
    });
    log('GCS', `Upload concluído: ${gcsUri}`);

    log('SpeechAPI', `Iniciando 'longRunningRecognize' para ${gcsUri}`);
    const [operation] = await speechClient.longRunningRecognize({
      audio: { uri: gcsUri },
      config: {
        encoding: 'WEBM_OPUS',
        sampleRateHertz: 48000,
        languageCode: 'pt-BR',
        enableAutomaticPunctuation: true,
        diarizationConfig: {
          enableSpeakerDiarization: true,
          minSpeakerCount: 2,
          maxSpeakerCount: 6,
        },
      },
    });

    const [response] = await operation.promise();
    log('SpeechAPI', `'longRunningRecognize' concluído para ${gcsUri}`);

    const structuredTranscript = [];
    let currentSegment = null;
    let lastSpeakerTag = null;

    response.results.forEach(result => {
      const alternative = result.alternatives[0];
      const transcriptText = alternative?.transcript?.trim();
      const words = alternative?.words;

      if (!transcriptText || !words || words.length === 0) return;

      const firstWord = words[0];
      const speakerTag = firstWord.speakerTag;

      if (speakerTag !== lastSpeakerTag) {
        currentSegment = {
          text: transcriptText,
          isFinal: true,
          speakerTag: speakerTag,
          timestamp: firstWord.startTime.seconds ? new Date(firstWord.startTime.seconds * 1000).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' }) : '00:00',
        };
        structuredTranscript.push(currentSegment);
        lastSpeakerTag = speakerTag;
      } else if (currentSegment) {
        currentSegment.text += ' ' + transcriptText;
      }
    });

    log('API', `Transcrição em lote estruturada com ${structuredTranscript.length} segmentos.`);
    res.json({ recordingId, audioUri: gcsUri, batchTranscript: structuredTranscript });

  } catch (err) {
    log('API-ERROR', `Erro em ${endpointName}:`, err);
    res.status(500).json({ error: 'Falha na transcrição em lote.' });
  }
});


// ==========================
// --- Endpoints Vertex AI ---
// ==========================

// --- GERAÇÃO DE TÍTULO ---
app.post('/api/generate-title', async (req, res) => {
  const endpointName = '/api/generate-title';
  try {
    const { context } = req.body;
    if (!context || typeof context !== 'string' || context.trim() === '') {
      return res.status(400).json({ error: 'O campo "context" é obrigatório.' });
    }

    const prompt = `
      Você é um assistente especializado em criar títulos curtos e objetivos para consultas médicas.
      Baseado no contexto abaixo, gere um título conciso (máx. 10 palavras) que resuma o motivo principal da consulta.
      O título deve ser claro, direto e fácil de entender. Não use markdown (como **, #) na resposta.

      Contexto: "${context}"

      Título Gerado:
    `;

    const generatedTitle = await callVertexAI(endpointName, prompt);
    res.status(200).json({ title: generatedTitle });

  } catch (error) {
    log('API-ERROR', `Erro em ${endpointName}:`, error);
    res.status(500).json({ error: 'Ocorreu um erro no servidor ao gerar o título.' });
  }
});


// --- MELHORAR ANAMNESE ---
app.post('/api/melhorar-anamnese', async (req, res) => {
    const endpointName = '/api/melhorar-anamnese';
    try {
        const { anamnese, prompt } = req.body;

        if (!anamnese || typeof anamnese !== 'string' || anamnese.trim() === '') {
            return res.status(400).json({ error: 'O campo "anamnese" é obrigatório.' });
        }
        if (!prompt || typeof prompt !== 'string' || prompt.trim() === '') {
            return res.status(400).json({ error: 'O campo "prompt" (instrução) é obrigatório.' });
        }

        const structuredPrompt = `
            ### Persona
            Aja como um assistente médico redator, especialista em criar documentos clínicos claros, objetivos e bem estruturados.

            ### Contexto
            O texto de uma anamnese médica precisa ser refinado com base em uma instrução específica do médico.

            ### Tarefa
            Reescreva o "Texto Original da Anamnese" abaixo, seguindo estritamente a "Instrução do Médico".

            ### Requisitos
            - O formato da resposta DEVE ser um único bloco de texto usando tags HTML simples (<p>, <strong>, <ul>, <li>).
            - O tom deve ser formal, técnico e objetivo.
            - Mantenha TODAS as informações clínicas originais. NÃO omita e NÃO invente dados.
            - Corrija erros gramaticais.

            ### Dados de Entrada
            **Instrução do Médico:**
            """
            ${prompt}
            """

            **Texto Original da Anamnese:**
            """
            ${anamnese}
            """
        `;

        const enhancedAnamnese = await callVertexAI(endpointName, structuredPrompt);
        res.status(200).json({ enhancedAnamnese });

    } catch (error) {
        log('API-ERROR', `Erro em ${endpointName}:`, error);
        res.status(500).json({ error: 'Ocorreu um erro no servidor ao processar a solicitação.' });
    }
});

// --- ROTA DE TRANSCRIÇÃO IA (CORRIGE E IDENTIFICA FALANTES) ---
app.post('/api/generate-ia-transcription', async (req, res) => {
  const endpointName = '/api/generate-ia-transcription';
  try {
    const { transcription } = req.body;
    if (!transcription || typeof transcription !== 'string' || transcription.trim() === '') {
      return res.status(400).json({ error: 'O campo "transcription" é obrigatório.' });
    }

    const prompt = `
      # Papel e Objetivo
      Você é um assistente de IA especialista em processar transcrições de consultas médicas. Sua tarefa é analisar o diálogo, identificar "Médico" e "Paciente", corrigir erros gramaticais e estruturar a saída em um JSON válido.

      # Instruções
      1.  **Identificação de Papéis**: Analise a conversa para determinar quem é o "Médico" (conduz a consulta) e o "Paciente" (descreve sintomas). Atribua os speakerTag 1 e 2 consistentemente a esses papéis.
      2.  **Aprimoramento do Texto**: Corrija erros gramaticais, de digitação e pontuação, mantendo o significado original.
      3.  **Formato de Saída OBRIGATÓRIO**:
          * Sua resposta deve ser um array de objetos JSON, um para cada fala.
          * Cada objeto deve conter EXATAMENTE os seguintes campos: speakerTag (string: "Médico" ou "Paciente"), text (string), e timestamp (string).
          * NÃO inclua NENHUM texto fora do array JSON. Sua resposta deve ser apenas o JSON puro.

      # Transcrição de Entrada
      """
      ${transcription}
      """
    `;

    const rawResponse = await callVertexAI(endpointName, prompt, { maxOutputTokens: 4096 });
    
    // Limpeza e parsing seguro da resposta da IA
    let parsedJson;
    try {
        const cleanedString = rawResponse.replace(/^```json\s*/, "").replace(/\s*```$/, "");
        parsedJson = JSON.parse(cleanedString);
        log('API', `JSON da transcrição IA foi parseado com sucesso para ${endpointName}.`);
    } catch (parseError) {
        log('API-ERROR', `Falha ao fazer parse do JSON retornado pela IA em ${endpointName}. Resposta crua:`, rawResponse);
        throw new Error("A resposta da IA não é um JSON válido.");
    }

    res.status(200).json({ title: parsedJson });

  } catch (error) {
    log('API-ERROR', `Erro em ${endpointName}:`, error);
    res.status(500).json({ error: 'Ocorreu um erro no servidor ao processar a transcrição.' });
  }
});


// --- GERAÇÃO DE RESUMO ---
app.post('/api/generate-summary', async (req, res) => {
  const endpointName = '/api/generate-summary';
  try {
    const { transcription } = req.body;
    if (!transcription || !Array.isArray(transcription) || transcription.length === 0) {
      return res.status(400).json({ error: 'O campo "transcription" é obrigatório e deve ser um array.' });
    }

    const formattedTranscription = transcription.map(item => `${item.speakerTag || 'Pessoa'}: ${item.text}`).join('\n');

    const prompt = `
      Você é um assistente de IA focado em transcrições médicas. Sua tarefa é gerar dois resultados claros, sem usar markdown ou introduções.

      1. **Resumo da Transcrição**: Crie um resumo objetivo da consulta.
      2. **Avaliação da Transcrição**:
         - Comente se a transcrição contém informações suficientes e coerentes.
         - Aponte lacunas ou inconsistências.
         - Avalie se faz sentido, no contexto médico, usar IA para gerar resumos desta transcrição.

      A transcrição é a seguinte:
      "${formattedTranscription}"
    `;

    const generatedSummary = await callVertexAI(endpointName, prompt);
    res.status(200).json({ summary: generatedSummary });

  } catch (error) {
    log('API-ERROR', `Erro em ${endpointName}:`, error);
    res.status(500).json({ error: 'Ocorreu um erro no servidor ao gerar o resumo.' });
  }
});

// --- GERAÇÃO DE ANAMNESE ---
app.post('/api/generate-anamnese', async (req, res) => {
  const endpointName = '/api/generate-anamnese';
  try {
    const { transcription, prompt, documentoSelecionado } = req.body;

    if (!transcription || !Array.isArray(transcription) || transcription.length === 0 || !documentoSelecionado) {
      return res.status(400).json({ error: 'Campos obrigatórios: transcription, documentoSelecionado.' });
    }

    const formattedTranscript = transcription.map(line => `${line.speakerTag}: ${line.text}`).join('\n');

    const fullPrompt = `
      Você é um assistente médico virtual que sumariza conversas clínicas em anamneses estruturadas.
      Sua tarefa é analisar a transcrição de uma consulta e gerar uma anamnese completa.

      Instruções Adicionais:
      ${prompt ? ` - Contexto do Paciente: "${prompt}"` : ''}
      
      O Documento deve conter as seguintes seções obrigatórias:
      ${documentoSelecionado}

      Formate o resultado em um único bloco de texto usando HTML (parágrafos, negrito, listas).

      Transcrição da consulta:
      "${formattedTranscript}"

      Anamnese Gerada (formato HTML):
    `;

    const generatedAnamnese = await callVertexAI(endpointName, fullPrompt);
    res.status(200).json({ anamnese: generatedAnamnese });

  } catch (error) {
    log('API-ERROR', `Erro em ${endpointName}:`, error);
    res.status(500).json({ error: 'Ocorreu um erro no servidor ao gerar a anamnese.' });
  }
});


// --- OBTENÇÃO DE URL DE ÁUDIO ---
app.get('/audio-url/:recordingId', async (req, res) => {
  const endpointName = '/audio-url/:recordingId';
  try {
    const bucketName = process.env.GCLOUD_BUCKET_NAME;
    const { recordingId } = req.params;
    const filename = `audio-${recordingId}.opus`;

    log('GCS', `Buscando URL assinada para ${filename}`);
    const file = storage.bucket(bucketName).file(filename);

    const [exists] = await file.exists();
    if (!exists) {
      log('GCS-ERROR', `Arquivo não encontrado: ${filename}`);
      return res.status(404).json({ error: 'Arquivo de áudio não encontrado.' });
    }

    const [signedUrl] = await file.getSignedUrl({
      action: 'read',
      expires: Date.now() + 15 * 60 * 1000, // 15 minutos
    });

    log('GCS', `URL assinada gerada com sucesso.`);
    res.json({ audioUrl: signedUrl });

  } catch (err) {
    log('API-ERROR', `Erro em ${endpointName}:`, err);
    res.status(500).json({ error: 'Falha ao gerar URL de áudio.' });
  }
});


// --- Iniciar Servidor ---
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  log('Server', `🚀 Servidor rodando na porta ${PORT}`);
});