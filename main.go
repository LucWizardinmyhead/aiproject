package main

import (
	"log"
	"net/http"
)

func main() {
	// Initialize routes
	StartScalingManager()
	http.HandleFunc("/v1/generate", generateHandler)
	http.HandleFunc("/models", listModelsHandler)
	http.HandleFunc("/health", healthHandler)

	log.Println("Starting Ollama Orchestrator on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
