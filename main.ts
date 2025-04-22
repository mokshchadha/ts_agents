import {GoogleAIAgent} from './src/google-ai-agent.ts'
import process from "node:process";

const service = new GoogleAIAgent(process.env.GEMINI_KEY!)

const clientCheckAgent = service.agent({
  name: 'ClientCheckAgent',
  instruction: "",
  description: "",
  isGoogleSearchEnabled: false
})