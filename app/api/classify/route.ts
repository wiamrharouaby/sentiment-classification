import { type NextRequest, NextResponse } from "next/server"

// Simuler une classification avec SVM
// Dans un environnement réel, cela appellerait notre API Python
export async function POST(request: NextRequest) {
  try {
    const { text } = await request.json()

    if (!text || typeof text !== "string") {
      return NextResponse.json({ error: "Le texte est requis" }, { status: 400 })
    }

    // Simulation d'une classification
    // Dans un environnement réel, nous appellerions notre API Python
    const isPositive = Math.random() > 0.5
    const confidence = 0.5 + Math.random() * 0.5

    // Attendre pour simuler le traitement
    await new Promise((resolve) => setTimeout(resolve, 1000))

    return NextResponse.json({
      sentiment: isPositive ? "positive" : "negative",
      confidence,
    })
  } catch (error) {
    console.error("Erreur de classification:", error)
    return NextResponse.json({ error: "Erreur lors de la classification" }, { status: 500 })
  }
}
