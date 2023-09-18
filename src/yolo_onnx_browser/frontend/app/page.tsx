/* eslint-disable @next/next/no-img-element */
"use client";
import { useEffect, useRef, useState } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import { detectImage } from "@/lib/utils";

const modelConfig = {
  name: "yolov8n.onnx",
  nmsModel: "nms-yolov8.onnx",
  inputShape: [1, 3, 640, 640],
  topK: 100,
  iouThreshold: 0.45,
  scoreThreshold: 0.25,
};

export default function Home() {
  const [session, setSession] = useState<{
    net: InferenceSession;
    nms: InferenceSession;
  } | null>(null);

  const [loading, setLoading] = useState(true);
  const [image, setImage] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputImageRef = useRef<HTMLImageElement>(null);
  const canvasOutputRef = useRef<HTMLCanvasElement>(null);
  const canvasInputRef = useRef<HTMLCanvasElement>(null);

  cv["onRuntimeInitialized"] = async () => {
    // create the YOLOv8 Model
    const yolov8 = await InferenceSession.create(`/model/${modelConfig.name}`, {
      executionProviders: ["wasm"],
    });

    // create the NMS Model
    const nms = await InferenceSession.create(
      `/model/${modelConfig.nmsModel}`,
      {
        executionProviders: ["wasm"],
      }
    );

    const tensor = new Tensor(
      "float32",
      new Float32Array(modelConfig.inputShape.reduce((a, b) => a * b)),
      modelConfig.inputShape
    );
    const res = await yolov8.run({ images: tensor });
    console.log("model warm up", res);

    setSession({
      net: yolov8,
      nms: nms,
    });

    setLoading(false);
  };
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-3xl">YOLOV8 - ONNX - WASM</h1>
      {loading && <>Loading Model...</>}
      <img
        ref={inputImageRef}
        src="#"
        alt=""
        // style={{ display: image ? "block" : "none" }}
        className="hidden absolute"
        onLoad={() => {
          if (!inputImageRef.current || !canvasOutputRef.current) return;
          if (!session) return;
          detectImage(
            inputImageRef.current,
            canvasOutputRef.current,
            session,
            modelConfig.topK,
            modelConfig.iouThreshold,
            modelConfig.scoreThreshold,
            modelConfig.inputShape
          );
        }}
      />
      <div className="relative min-h-[640px] min-w-[640px]">
        <div className="absolute flex flex-col items-center w-full justify-center z-20">
          <canvas
            width={modelConfig.inputShape[2]}
            height={modelConfig.inputShape[3]}
            ref={canvasInputRef}
            className="absolute left-0 top-0 rounded-md"
          />
          <canvas
            width={modelConfig.inputShape[2]}
            height={modelConfig.inputShape[3]}
            ref={canvasOutputRef}
            className="absolute left-0 top-0"
          />
        </div>
      </div>
      <input
        type="file"
        ref={inputRef}
        accept="image/*"
        onChange={(e) => {
          if (!inputImageRef.current) return;
          if (e.target.files?.length) {
            // handle next image to detect
            if (image) {
              URL.revokeObjectURL(image);
              setImage(null);
            }

            const url = URL.createObjectURL(e.target.files[0]); // create image url
            inputImageRef.current.src = url; // set image source

            const canvas2DCtx = canvasInputRef.current?.getContext("2d");

            inputImageRef.current.onload = async () => {
              if (!inputImageRef.current) return;
              if (canvas2DCtx) {
                canvas2DCtx.drawImage(
                  inputImageRef.current,
                  0,
                  0,
                  modelConfig.inputShape[2],
                  modelConfig.inputShape[3]
                );
              }
            };

            setImage(url);
          }
        }}
      />
    </main>
  );
}
