"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"
import { Loader2, CheckCircle, XCircle } from "lucide-react"

export default function Home() {
  const [text, setText] = useState("")
  const [prediction, setPrediction] = useState<null | { sentiment: string; confidence: number }>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const classifyText = async () => {
    if (!text.trim()) {
      setError("Veuillez entrer un texte à classifier")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error("Erreur lors de la classification")
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError("Une erreur est survenue lors de la classification")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-50">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Classification de Texte MLOps</CardTitle>
          <CardDescription>Entrez un texte pour le classifier comme positif ou négatif</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Entrez votre texte ici..."
            className="min-h-32"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          {error && <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">{error}</div>}

          {prediction && (
            <Card
              className={`border ${prediction.sentiment === "positive" ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}
            >
              <CardContent className="pt-4">
                <div className="flex items-center gap-2">
                  {prediction.sentiment === "positive" ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-600" />
                  )}
                  <span className="font-medium">
                    Sentiment: {prediction.sentiment === "positive" ? "Positif" : "Négatif"}
                  </span>
                </div>
                <div className="mt-2">
                  <span className="text-sm text-gray-600">Confiance: {(prediction.confidence * 100).toFixed(2)}%</span>
                </div>
              </CardContent>
            </Card>
          )}
        </CardContent>
        <CardFooter>
          <Button onClick={classifyText} disabled={loading} className="w-full">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Classification en cours...
              </>
            ) : (
              "Classifier"
            )}
          </Button>
        </CardFooter>
      </Card>
    </main>
  )
}
