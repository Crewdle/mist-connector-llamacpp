import { EventEmitter } from 'events';

import type { Llama, LlamaModel, LlamaEmbeddingContext, LlamaContext } from 'node-llama-cpp';

import type { GenerativeAIModelOutputType, IGenerativeAIWorkerConnector, IGenerativeAIWorkerOptions, IJobParametersAI, IJobResultAI } from '@crewdle/web-sdk-types';

import { ILlamacppGenerativeAIWorkerOptions } from './LlamacppGenerativeAIWorkerOptions';
import { ILlamacppGenerativeAIWorkerModel } from './LlamacppGenerativeAIWorkerModel';

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
   * The workflow ID.
   * @ignore
   */
  private workflowId?: string;

  /**
   * The context.
   * @ignore
   */
  private context?: {
    modelId: string;
    instance: LlamaContext;
  }

  /**
   * The embedding context.
   * @ignore
   */
  private embeddingContext?: {
    modelId: string;
    instance: LlamaEmbeddingContext;
  }

  /**
   * The Llama engine.
   * @ignore
   */
  private static engine?: Llama;

  /**
   * The models.
   * @ignore
   */
  private static models: Map<string, ILlamacppGenerativeAIWorkerModel> = new Map();

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
   * Get the Llama engine.
   * @returns A promise that resolves with the Llama engine.
   * @ignore
   */
  private static async getEngine(): Promise<Llama> {
    if (this.engine) {
      return this.engine;
    }

    const { getLlama } = await import('node-llama-cpp');
    this.engine = await getLlama();
    return this.engine;
  }

  /**
   * Initialize the machine learning model.
   * @param workflowId The workflow ID.
   * @param models The models to initialize.
   */
  async initialize(workflowId: string, models: Map<string, string>): Promise<void> {
    const engine = await LlamacppGenerativeAIWorkerConnector.getEngine();
    for (const [modelName, modelPath] of models) {
      if (!LlamacppGenerativeAIWorkerConnector.models.has(modelName)) {
        LlamacppGenerativeAIWorkerConnector.models.set(modelName, {
          model: await engine.loadModel({ modelPath }),
          workflows: new Set(),
        });
      }
      const model = LlamacppGenerativeAIWorkerConnector.models.get(modelName);
      if (model) {
        model.workflows.add(workflowId);
      }
      this.workflowId = workflowId;
    }
  }

  /**
   * Close the machine learning model.
   * @returns A promise that resolves when the model has been closed.
   */
  async close(): Promise<void> {
    if (!this.workflowId) {
      return;
    }
    for (const [id, model] of LlamacppGenerativeAIWorkerConnector.models) {
      model.workflows.delete(this.workflowId);
      LlamacppGenerativeAIWorkerConnector.models.set(id, model);
      if (model.workflows.size === 0) {
        await model.model.dispose();
        LlamacppGenerativeAIWorkerConnector.models.delete(id);
      }
    }
  }

  /**
   * Process a job.
   * @param parameters The job parameters.
   * @returns A promise that resolves with the job result.
   */
  async processJob(parameters: IJobParametersAI, options: IGenerativeAIWorkerOptions): Promise<IJobResultAI> {
    const model = LlamacppGenerativeAIWorkerConnector.models.get(options.model.id)?.model;
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      if (!this.embeddingContext || this.embeddingContext.modelId !== options.model.id) {
        if (this.embeddingContext) {
          await this.embeddingContext.instance.dispose();
        }
        this.embeddingContext = {
          modelId: options.model.id,
          instance: await model.createEmbeddingContext(),
        };
      }
      const vector = await this.getVector(this.embeddingContext.instance, parameters.prompt);
      return {
        output: vector,
      };
    }
    
    if (!this.context || this.context.modelId !== options.model.id) {
      if (this.context) {
        await this.context.instance.dispose();
      }
      this.context = {
        modelId: options.model.id,
        instance: await model.createContext(),
      };
    }
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: this.context.instance.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const output = await session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
    });

    const inputTokens = model.tokenize(prompt).length;
    const outputTokens = model.tokenize(output).length;
    
    session.dispose();

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
    const model = LlamacppGenerativeAIWorkerConnector.models.get(options.model.id)?.model;
    if (!model) {
      throw new Error('Model not initialized');
    }

    if (options.model.outputType === 'vector' as GenerativeAIModelOutputType.Vector) {
      throw new Error('Vector output type not supported for streaming');
    }
    
    if (!this.context || this.context.modelId !== options.model.id) {
      if (this.context) {
        await this.context.instance.dispose();
      }
      this.context = {
        modelId: options.model.id,
        instance: await model.createContext(),
      };
    }
    const { LlamaChatSession } = await import('node-llama-cpp');
    const session = new LlamaChatSession({
      contextSequence: this.context.instance.getSequence(),
    });

    const prompt = await this.getPrompt(parameters, model);

    const inputTokens = model.tokenize(prompt).length;
    let outputTokens = 0;

    const textEmitter = new EventEmitter();

    session.prompt(prompt, {
      maxTokens: parameters.maxTokens ?? this.maxTokens,
      temperature: parameters.temperature ?? this.temperature,
      onTextChunk: (text) => {
        textEmitter.emit('text', text);
      },
    }).then(() => {
      textEmitter.emit('text', undefined);
    }).catch(e => {});

    while (true) {
      const text = await new Promise<string>((resolve) => textEmitter.once('text', resolve));

      if (text === undefined) {
        break;
      }

      outputTokens += model.tokenize(text).length;
      yield {
        output: text,
        inputTokens,
        outputTokens,
      };
    }

    session.dispose();
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
