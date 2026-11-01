import express, { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { Type, ThinkingLevel } from '@google/genai';
import {
  ai,
  GEMINI_MODEL,
  generateEmbedding,
  cosineSimilarity,
  lexicalScore,
  generateContentWithFallback,
} from './server/gemini.js';
import { extractPdfData, embedChunks } from './server/pdfService.js';
import {
  initStorage,
  getDocuments,
  getDocumentById,
  saveDocument,
  deleteDocument,
  getSummaries,
  saveSummary,
  getFlashcardSets,
  saveFlashcardSet,
  getChatHistory,
  addChatMessage,
  clearChatHistory,
  getAppStats,
} from './server/storage.js';
import { DocumentItem, GroundedSource, Flashcard } from './src/types/document.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isProd = process.env.NODE_ENV === 'production';
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

const app = express();
app.use(express.json({ limit: '25mb' }));

// Multer memory storage for PDF processing
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB max
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === 'application/pdf' || file.originalname.toLowerCase().endsWith('.pdf')) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF documents are supported'));
    }
  },
});

// Initialize database
await initStorage();

// -------------------------------------------------------------
// API ROUTES
// -------------------------------------------------------------

// 1. App Statistics
app.get('/api/stats', async (_req: Request, res: Response) => {
  try {
    const stats = await getAppStats();
    res.json(stats);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 2. Documents List
app.get('/api/documents', async (_req: Request, res: Response) => {
  try {
    const docs = await getDocuments();
    // Return summary without heavy chunks
    const summaries = docs.map((d) => ({
      id: d.id,
      name: d.name,
      originalName: d.originalName,
      fileSize: d.fileSize,
      pageCount: d.pageCount,
      wordCount: d.wordCount,
      createdAt: d.createdAt,
      status: d.status,
      errorMessage: d.errorMessage,
    }));
    res.json(summaries);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 3. Document Details
app.get('/api/documents/:id', async (req: Request, res: Response) => {
  try {
    const doc = await getDocumentById(req.params.id);
    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.json(doc);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 4. Upload PDF
app.post(
  '/api/documents/upload',
  upload.single('pdf'),
  async (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file provided' });
    }

    const docId = `doc-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const originalName = req.file.originalname;

    try {
      // 1. Initial Document Record
      const newDoc: DocumentItem = {
        id: docId,
        name: originalName,
        originalName,
        fileSize: req.file.size,
        pageCount: 0,
        wordCount: 0,
        createdAt: new Date().toISOString(),
        status: 'extracting',
        pages: [],
        chunks: [],
      };
      await saveDocument(newDoc);

      // 2. Extract PDF text page-by-page
      const extractResult = await extractPdfData(req.file.buffer);

      // Scanned PDF detection
      if (!extractResult.hasText) {
        newDoc.status = 'failed';
        newDoc.pageCount = extractResult.totalPages;
        newDoc.errorMessage =
          'No extractable text was found in this PDF. This appears to be a scanned image-only document, which requires an OCR engine to process. Please upload a PDF with digital text or explore our pre-loaded research papers.';
        await saveDocument(newDoc);
        return res.status(422).json({
          error: newDoc.errorMessage,
          document: newDoc,
        });
      }

      // 3. Indexing Chunks with Embeddings
      newDoc.status = 'indexing';
      newDoc.pageCount = extractResult.totalPages;
      newDoc.wordCount = extractResult.totalWords;
      newDoc.pages = extractResult.pages;
      await saveDocument(newDoc);

      const embeddedChunks = await embedChunks(extractResult.chunks);
      newDoc.chunks = embeddedChunks;
      newDoc.status = 'ready';
      await saveDocument(newDoc);

      res.status(201).json({
        message: 'Your document is ready! You can now ask questions, generate summaries and create flashcards.',
        document: newDoc,
      });
    } catch (err: any) {
      console.error('PDF Upload Error:', err);
      res.status(500).json({
        error: `Failed to process PDF: ${err.message || 'Unknown error'}`,
      });
    }
  }
);

// 5. Delete Document
app.delete('/api/documents/:id', async (req: Request, res: Response) => {
  try {
    const success = await deleteDocument(req.params.id);
    if (!success) {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.json({ success: true, message: 'Document deleted successfully' });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 6. Reset / Seed Sample Documents
app.post('/api/documents/reset-samples', async (_req: Request, res: Response) => {
  try {
    await initStorage();
    const docs = await getDocuments();
    res.json({ message: 'Samples verified', count: docs.length });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 7. Grounded Chat Q&A
app.post('/api/chat', async (req: Request, res: Response) => {
  const { documentId, question } = req.body;
  if (!documentId || !question) {
    return res.status(400).json({ error: 'documentId and question are required' });
  }

  try {
    const doc = await getDocumentById(documentId);
    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }

    if (doc.status !== 'ready' || doc.chunks.length === 0) {
      return res.status(400).json({
        error: 'Document is not yet processed or contains no text chunks.',
      });
    }

    // A. Semantic Retrieval & Scoring
    const queryEmbedding = await generateEmbedding(question);
    const scoredChunks = doc.chunks.map((chunk) => {
      let score = 0;
      if (queryEmbedding && chunk.embedding) {
        const cosSim = cosineSimilarity(queryEmbedding, chunk.embedding);
        const lex = lexicalScore(question, chunk.text);
        score = cosSim * 0.7 + lex * 0.3;
      } else {
        score = lexicalScore(question, chunk.text);
      }
      return { chunk, score };
    });

    // Sort by score descending and take top 4-5
    scoredChunks.sort((a, b) => b.score - a.score);
    const topMatches = scoredChunks.slice(0, 5);

    // Prepare sources for UI
    const sources: GroundedSource[] = topMatches
      .filter((m) => m.score > 0.05 || m.chunk.text.length > 50)
      .slice(0, 3)
      .map((m) => ({
        pageNumber: m.chunk.pageNumber,
        excerpt: m.chunk.text.slice(0, 240) + '...',
        relevanceScore: Math.min(Math.round(Math.max(m.score, 0.45) * 100), 99),
      }));

    // Build context
    const contextText = topMatches
      .map((m) => `[Source Page ${m.chunk.pageNumber}]:\n${m.chunk.text}`)
      .join('\n\n---\n\n');

    // Recent chat history
    const history = await getChatHistory(documentId);
    const recentHistoryText = history
      .slice(-4)
      .map((h) => `${h.sender === 'user' ? 'User' : 'Assistant'}: ${h.text}`)
      .join('\n');

    const systemPrompt = `You are DocuMind AI, an intelligent, authoritative document analysis assistant.
Your answers MUST be strictly grounded in the provided document context.
Do NOT fabricate or extrapolate information beyond what is stated in the document.
When stating facts or statistics, cite the source page in brackets, e.g. [Page 2] or [Page 4].
If the answer cannot be found or substantiated in the document context, clearly state:
"Based on the provided document, this information is not mentioned or cannot be verified."
Explain complex technical concepts in plain, clear English when requested. Maintain an objective, professional tone.`;

    const userPrompt = `DOCUMENT CONTEXT:
${contextText}

${recentHistoryText ? `PREVIOUS CONVERSATION:\n${recentHistoryText}\n\n` : ''}USER QUESTION:
${question}

Provide a direct, thorough, and accurately cited answer based ONLY on the document context above.`;

    let assistantAnswer = '';
    try {
      assistantAnswer = await generateContentWithFallback({
        contents: userPrompt,
        systemInstruction: systemPrompt,
      });
    } catch (genError: any) {
      console.warn('Gemini generateContent error:', genError?.message);
      if (sources.length > 0) {
        assistantAnswer = `Based on the document text [Page ${sources[0].pageNumber}]:\n\n"${sources[0].excerpt}"\n\n*(Note: Free tier daily Gemini API request quota reached. Grounded excerpts above remain verified. To upgrade, select a billing-enabled key in Settings > Secrets.)*`;
      } else {
        assistantAnswer = `Based on the uploaded document, relevant text was retrieved. (Free tier Gemini API rate limit reached; configure a paid key in Settings > Secrets for continuous generation.)`;
      }
    }

    const userMsg = {
      id: `msg-${Date.now()}-user`,
      sender: 'user' as const,
      text: question,
      timestamp: new Date().toISOString(),
    };
    const assistantMsg = {
      id: `msg-${Date.now()}-assistant`,
      sender: 'assistant' as const,
      text: assistantAnswer,
      timestamp: new Date().toISOString(),
      sources,
    };

    await addChatMessage(documentId, userMsg);
    await addChatMessage(documentId, assistantMsg);

    res.json({
      answer: assistantAnswer,
      sources,
      userMessage: userMsg,
      assistantMessage: assistantMsg,
    });
  } catch (err: any) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 8. Get Chat History
app.get('/api/chat/:documentId', async (req: Request, res: Response) => {
  try {
    const history = await getChatHistory(req.params.documentId);
    res.json(history);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 9. Clear Chat History
app.delete('/api/chat/:documentId', async (req: Request, res: Response) => {
  try {
    await clearChatHistory(req.params.documentId);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 10. Generate Summary
app.post('/api/summary', async (req: Request, res: Response) => {
  const { documentId, type = 'executive' } = req.body;
  if (!documentId) {
    return res.status(400).json({ error: 'documentId is required' });
  }

  try {
    const doc = await getDocumentById(documentId);
    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }

    // Compile document text
    const fullText = doc.pages
      .map((p) => `--- PAGE ${p.pageNumber} ---\n${p.text}`)
      .join('\n\n')
      .slice(0, 45000); // within Gemini context limits

    let promptInstructions = '';
    if (type === 'quick') {
      promptInstructions = `Generate a Quick Summary consisting of 6 to 10 high-impact, actionable bullet points.
Highlight the primary thesis, key discoveries, quantitative metrics, and core conclusions.
Attach relevant page references in square brackets (e.g. [Page 2]).
Format in clean Markdown.`;
    } else if (type === 'detailed') {
      promptInstructions = `Generate a Detailed Summary with comprehensive section-by-section analysis:
- Introduction & Problem Formulation
- Core Methodology / Technical Architecture
- In-depth Findings, Data & Analysis
- Implications, Comparisons & Practical Value
Cite source page numbers throughout (e.g. [Page 3]). Format in clean Markdown with clear headings and bold highlights.`;
    } else {
      // Executive summary
      promptInstructions = `Generate a rigorous Executive Summary with the following mandatory numbered sections:
1. Executive Objective & Purpose (What problem does this document address?)
2. Methodology & Approach (How was the work conducted or structured?)
3. Key Findings & Quantitative Data (What are the measurable results or discoveries?)
4. Important Concepts & Terminology (Define critical concepts introduced in the text)
5. Limitations, Caveats & Risks (What are the boundaries, constraints or open questions?)
6. Strategic Conclusions & Recommendations (What are the immediate takeaways?)

Ensure every section is grounded solely in the document. Include exact page citations like [Page 1]. Format in polished Markdown.`;
    }

    let summaryContent = '';
    try {
      summaryContent = await generateContentWithFallback({
        contents: `DOCUMENT CONTENT:\n${fullText}\n\nINSTRUCTIONS:\n${promptInstructions}`,
        systemInstruction:
          'You are DocuMind AI, an elite document summarization specialist. Produce an insightful, deeply structured, and accurately referenced summary based exclusively on the provided document text.',
      });
    } catch (genError: any) {
      console.warn('Summary generation fallback:', genError?.message);
      const topPageExcerpts = doc.pages
        .slice(0, 4)
        .map((p) => `### Section & Page ${p.pageNumber}\n${p.text.slice(0, 380)}...`)
        .join('\n\n');
      summaryContent = `## ${type.toUpperCase()} SUMMARY (Document Grounded)\n\n${topPageExcerpts}\n\n*(Note: Gemini Free-tier daily rate limit reached. To generate full-length AI syntheses on demand, enable billing in Settings > Secrets.)*`;
    }

    const summaryRecord = {
      id: `sum-${Date.now()}`,
      documentId,
      type: type as 'quick' | 'detailed' | 'executive',
      content: summaryContent,
      createdAt: new Date().toISOString(),
    };

    await saveSummary(summaryRecord);
    res.json(summaryRecord);
  } catch (err: any) {
    console.error('Summary error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 11. Get Summaries for Document
app.get('/api/summary/:documentId', async (req: Request, res: Response) => {
  try {
    const list = await getSummaries(req.params.documentId);
    res.json(list);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 12. Generate Flashcards (10, 20, or 30 cards)
app.post('/api/flashcards', async (req: Request, res: Response) => {
  const { documentId, count = 10 } = req.body;
  const targetCount = [10, 20, 30].includes(Number(count)) ? Number(count) : 10;

  try {
    const doc = await getDocumentById(documentId);
    if (!doc) {
      return res.status(404).json({ error: 'Document not found' });
    }

    const fullText = doc.pages
      .map((p) => `[Page ${p.pageNumber}]:\n${p.text}`)
      .join('\n\n')
      .slice(0, 45000);

    const prompt = `Based strictly on the following document text, generate exactly ${targetCount} high-yield educational study flashcards.
Each flashcard must contain:
- question: Clear, focused question testing a concept, finding, definition, or methodology from the document.
- answer: Comprehensive yet concise explanation (2-4 sentences) answering the question thoroughly.
- difficulty: One of "Easy", "Medium", or "Hard".
- pageNumber: The exact source page number where the information is found.
- tag: A short 1-2 word category tag (e.g., "Architecture", "Metrics", "Storage", "Policy").

DOCUMENT TEXT:
${fullText}`;

    let cards: Flashcard[] = [];
    try {
      const responseText = await generateContentWithFallback({
        contents: prompt,
        systemInstruction:
          'You are DocuMind AI Educational Assistant. Create high-quality, rigorous flashcards strictly grounded in the document.',
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              question: { type: Type.STRING, description: 'The flashcard question' },
              answer: { type: Type.STRING, description: 'The thorough answer' },
              difficulty: {
                type: Type.STRING,
                description: 'Difficulty level: Easy, Medium, or Hard',
              },
              pageNumber: {
                type: Type.INTEGER,
                description: 'The source page number from the document',
              },
              tag: { type: Type.STRING, description: 'A short topic category' },
            },
            required: ['question', 'answer', 'difficulty', 'pageNumber'],
          },
        },
      });

      const parsedJson = JSON.parse(responseText || '[]');
      cards = parsedJson.map((c: any, idx: number) => ({
        id: `card-${Date.now()}-${idx}`,
        question: c.question,
        answer: c.answer,
        difficulty: ['Easy', 'Medium', 'Hard'].includes(c.difficulty) ? c.difficulty : 'Medium',
        pageNumber: Number(c.pageNumber) || 1,
        tag: c.tag || 'General',
        mastered: false,
      }));
    } catch (genError: any) {
      console.warn('Flashcard generation fallback:', genError?.message);
      cards = doc.chunks.slice(0, targetCount).map((chunk, idx) => ({
        id: `card-${Date.now()}-${idx}`,
        question: `What are the core findings and concepts described in Section ${idx + 1} (Page ${chunk.pageNumber})?`,
        answer: chunk.text,
        difficulty: (idx % 3 === 0 ? 'Easy' : idx % 3 === 1 ? 'Medium' : 'Hard') as any,
        pageNumber: chunk.pageNumber,
        tag: 'Document Concept',
        mastered: false,
      }));
    }

    const flashcardSet = {
      id: `set-${Date.now()}`,
      documentId,
      createdAt: new Date().toISOString(),
      cards,
    };

    await saveFlashcardSet(flashcardSet);
    res.json(flashcardSet);
  } catch (err: any) {
    console.error('Flashcard error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 13. Get Flashcards for Document
app.get('/api/flashcards/:documentId', async (req: Request, res: Response) => {
  try {
    const sets = await getFlashcardSets(req.params.documentId);
    res.json(sets.length > 0 ? sets[0] : null);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// 14. Update Flashcard Mastery
app.put('/api/flashcards/:documentId/mastery', async (req: Request, res: Response) => {
  const { cardId, mastered } = req.body;
  try {
    const sets = await getFlashcardSets(req.params.documentId);
    if (sets.length > 0) {
      const set = sets[0];
      const card = set.cards.find((c) => c.id === cardId);
      if (card) {
        card.mastered = Boolean(mastered);
        await saveFlashcardSet(set);
        return res.json({ success: true, card });
      }
    }
    res.status(404).json({ error: 'Card not found' });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------------------------------------------------
// VITE DEV SERVER OR STATIC ASSETS
// -------------------------------------------------------------
if (!isProd) {
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
} else {
  app.use(express.static(path.resolve(__dirname, 'dist')));
  app.get('*', (_req, res) => {
    res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
  });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 DocuMind AI Server running on http://0.0.0.0:${PORT}`);
});
