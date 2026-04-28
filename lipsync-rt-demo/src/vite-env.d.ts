/// <reference types="vite/client" />

type PuterTxt2SpeechOptions = {
  voice?: string;
  engine?: string;
  language?: string;
  provider?: string;
  model?: string;
  response_format?: string;
  test_mode?: boolean;
};

interface PuterGlobal {
  ai: {
    txt2speech(
      text: string,
      options?: PuterTxt2SpeechOptions
    ): Promise<HTMLAudioElement>;
  };
}

declare global {
  interface Window {
    puter?: PuterGlobal;
  }
}

export {};
