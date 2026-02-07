import { cos_sim, env, pipeline, TextStreamer } from '@huggingface/transformers'
import { createInterface } from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'
import { readFile } from './IOUtils'
env.backends.onnx.logLevel = 'error'

// GOOD LUCK!
