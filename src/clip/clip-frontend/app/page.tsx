"use client"
import { useState } from 'react'

export default function Home() {
  const [inputImage, setInputImage] = useState<File>();
  const [inputText, setInputText] = useState("");
  const [response, setResponse] = useState("");
  return (
    <main className="flex min-h-screen flex-col p-24">
      <h1>CLIPService</h1>
      <input 
        type="text" 
        className="text-black"
        value={inputText}
        onChange={(e) => {setInputText(e.target.value)}} 
      />
      <input 
        type="file" 
        accept="image/png, image/jpeg" 
        onChange={(e) => {
          if (!e.target.files) return;
          if (e.target?.files?.length > 0) {
            console.log(e.target.files)
            setInputImage(e.target.files[0])
          }
        }} 
      />
      <button onClick={() => {
        if (!inputImage || !inputText) return;

        let formdata = new FormData();
        formdata.append("text", inputText);
        formdata.append("file", inputImage, inputImage?.name);
        
        const requestOptions = {
          method: 'POST',
          body: formdata,
        };

        fetch(`http://34.196.83.58/image_to_text`, requestOptions)
          .then((response) => response.json())
          .then((result) => {
            setResponse(JSON.stringify(result));
          })
          .catch((error) => console.log("error", error));
      }}>submit</button>
      <pre>{response}</pre>
    </main>
  )
}