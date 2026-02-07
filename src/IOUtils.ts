import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export function readFile(file: string) {
  const inputPath = path.resolve(__dirname, file)
  return fs.readFileSync(inputPath, 'utf8')
}

export function writeFile(file: string, content: string) {
  fs.writeFileSync(path.resolve(__dirname, file), content, 'utf8')
}
