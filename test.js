// Si Node < 18, installer node-fetch : npm install node-fetch
import fetch from "node-fetch";
import { sentences } from "./sentences.js";



async function translateText(index, src, tgt, text) {
  const url = "http://localhost:8000/v1/chat/completions";

  const body = {
    model: "Infomaniak-AI/vllm-translategemma-4b-it",
    messages: [
      {
        role: "user",
        content: `<<<source>>>${src}<<<target>>>${tgt}<<<text>>>${text}`
      }
    ]
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    console.log(`Réponse:${index} => ${data.choices[0].message.content}`);
  } catch (err) {
    console.error("Erreur:", err);
  }
}

/*console.time('TOTAL')
for(const traduction of transcripts) {
    console.time('TIME')
    await translateText('en', 'fr',traduction);
    console.timeEnd('TIME')
}
console.timeEnd('TOTAL')*/



//PARALLEL
console.time('PARALLEL');
let id=0;
const results = await Promise.all(sentences.map((traduction) => 
    translateText(id++, 'en', 'fr', traduction )
));
console.log('END : ', results.length);
console.timeEnd('PARALLEL');

/*
for(const traduction of traductions) {
    console.time('TIME')
    await translateText(traduction.src_language, traduction.tgt_language,traduction.text );
    console.timeEnd('TIME')
}*/