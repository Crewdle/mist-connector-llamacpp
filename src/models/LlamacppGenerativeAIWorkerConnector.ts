import { EventEmitter } from 'events';

import type { Llama, LlamaModel, LlamaEmbeddingContext, Token } from 'node-llama-cpp';

import type { GenerativeAIModelOutputType, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, IJobParametersAI, IJobResultAI } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';

/**
 * The Llamacpp machine learning connector.
 */
export class LlamacppGenerativeAIWorkerConnector implements IGenerativeAIWorkerConnector {
  /**
   * The instructions.
   * @ignore
   */
  private instructions = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it will use concepts stored in memory that have very similar weights.`;

  /**
   * The maximum number of tokens to generate.
   * @ignore
   */
  private maxTokens = 1024;

  /**
   * The temperature.
   * @ignore
   */
  private temperature = 1;

  /**
   * The Llama engine.
   * @ignore
   */
  private engine?: Llama;

  /**
   * The models.
   */
  private models: Map<string, LlamaModel> = new Map();

  /**
   * The constructor.
   * @param options The options.
   */
  constructor(
    private options?: ILlamacppGenerativeAIWorkerOptions,
  ) {
    if (this.options?.instructions) {
      this.instructions = this.options.instructions;
    }
    if (this.options?.maxTokens) {
      this.maxTokens = this.options.maxTokens;
    }
    if (this.options?.temperature) {
      this.temperature = this.options.temperature;
    }
  }

  /**
   * Initialize the machine learning model.
   * @param models The models to initialize.
   */
  async initialize(models: Map<string, string>): Promise<void> {
    const { getLlama } = await import('node-llama-cpp');
    this.engine = await getLlama();
    for (const [modelName, modelPath] of models) {
      this.models.set(modelName, await this.engine.loadModel({ modelPath }));
    }
  }

  /**
   * Process a job.
   * @param parameters The job parameters.
   * @returns A promise that resolves with the job result.
   */
  async processJob(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): Promise<IJobResultAI> {
    const model = this.models.get(options.model.id);
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      const context = await model.createEmbeddingContext();
      const vector = await this.getVector(context, parameters.prompt);
      await context.dispose();
      return {
        output: vector,
      };
    }
    
    const context = await model.createContext();
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const output = await session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
    });

    const inputTokens = model.tokenize(prompt).length;
    const outputTokens = model.tokenize(output).length;
    
    session.dispose();
    await context.dispose();

    return {
      output,
      inputTokens,
      outputTokens,
    };
  }

  /**
   * Stream a job.
   * @param parameters The job parameters.
   * @returns An async generator that yields the responses.
   */
  async *processJobStream(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): AsyncGenerator<IJobResultAI> {
    const model = this.models.get(options.model.id);
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      throw new Error('Vector output type not supported for streaming');
    }
    
    const context = await model.createContext();
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const inputTokens = model.tokenize(prompt).length;
    let outputTokens = 0;

    const tokenEmitter = new EventEmitter();

    session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
      onToken: async (token) => {
        tokenEmitter.emit('token', token);
      }
    }).then(() => {
      tokenEmitter.emit('token', undefined);
    }).catch(e => {});

    while (true) {
      const token = await new Promise<Token[]>((resolve) => tokenEmitter.once('token', resolve));
      if (token === undefined) {
        break;
      }
      const output = model.detokenize(token);
      if (output.indexOf('<|end|>') !== -1) {
        break;
      }
      outputTokens += token.length;
      yield {
        output,
        inputTokens,
        outputTokens,
      };
    }

    session.dispose();
    await context.dispose();
  }

  private async getPrompt(parameters: IJobParametersAI, model: LlamaModel): Promise<string> {
    const { prompt, instructions, history } = parameters;

    let finalPrompt = `${instructions ?? this.instructions}\n\n`;

    finalPrompt += `Conversation:\n`;
    if (history) {
      let conversation = '';
      let conversationLength = model.tokenize(finalPrompt).length + model.tokenize(`Human: ${prompt}\nAI:`).length;
      for (let i = history.length - 1; i >= 0; i--) {
        const { source, message } = history[i];
        if (conversationLength + model.tokenize(message).length > model.trainContextSize * 0.75) {
          break;
        }
        conversation = `${source === 'ai' ? 'AI' : 'Human'}: ${message}\n${conversation}`;
        conversationLength += model.tokenize(`${source === 'ai' ? 'AI' : 'Human'}: ${message}\n`).length;
      }
      finalPrompt += conversation;
    }

    finalPrompt += `Human: ${prompt}\nAI:`;

    return finalPrompt;
  }

  /**
   * Get the vector for some content.
   * @param embeddingContext The embedding context.
   * @param content The content.
   * @returns A promise that resolves to the vector.
   * @ignore
   */
  private async getVector(embeddingContext: LlamaEmbeddingContext, content: string): Promise<number[]> {
    const vector = (await embeddingContext.getEmbeddingFor(this.cleanText(content))).vector;
    return this.normalizeVector(vector);
  }

  /**
   * Clean the text.
   * @param text The text to clean.
   * @returns The cleaned text.
   * @ignore
   */
  private cleanText(text: string): string {
    return text.trim().toLowerCase().replace(/[^a-z0-9\s]/g, '');
  }

  /**
   * Normalize a vector.
   * @param vector The vector to normalize.
   * @returns The normalized vector.
   * @ignore
   */
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return vector.map((value) => value / norm);
  }
}
