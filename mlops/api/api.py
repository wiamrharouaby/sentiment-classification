from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import jwt
from datetime import datetime, timedelta
import bcrypt
import logging
from functools import wraps
import time
from mlops.models.model_singleton import ModelSingleton

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Décorateur pour mesurer le temps d'exécution des endpoints
def timing_endpoint(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Endpoint {func.__name__} exécuté en {end_time - start_time:.4f} secondes")
        return result
    return wrapper


# Modèles Pydantic pour la validation des données
class TextRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    sentiment: str
    confidence: float


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserRegistration(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None


# Configuration de l'API
app = FastAPI(
    title="API de Classification de Texte",
    description="API pour la classification de sentiment de texte",
    version="1.0.0",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de la sécurité
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "votre_clé_secrète_à_changer"  # À changer en production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Middleware pour logger les requêtes (sans lire le corps : cela bloquait les POST)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Requête reçue: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Réponse envoyée: {response.status_code}")
    return response


# Fonctions d'authentification
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# Base de données fictive d'utilisateurs (en mémoire)
fake_users_db: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "full_name": "Administrateur",
        "email": "admin@example.com",
        "hashed_password": hash_password("admin"),
        "disabled": False,
    },
    "user": {
        "username": "user",
        "full_name": "Utilisateur Normal",
        "email": "user@example.com",
        "hashed_password": hash_password("user"),
        "disabled": False,
    },
}


def get_user(db, username: str) -> Optional[User]:
    if username in db:
        return User(**db[username])
    return None


def authenticate_user(db, username: str, password: str) -> Optional[User]:
    logger.info(f"Tentative d'authentification pour l'utilisateur: {username}")

    if username not in db:
        logger.warning(f"Utilisateur {username} non trouvé dans la base de données")
        return None

    user_dict = db[username]
    if not verify_password(password, user_dict["hashed_password"]):
        logger.warning(f"Mot de passe incorrect pour l'utilisateur {username}")
        return None

    logger.info(f"Authentification réussie pour l'utilisateur {username}")
    return User(**user_dict)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les informations d'identification",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user


# Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", response_model=Dict[str, str])
async def register_user(user_data: UserRegistration):
    if user_data.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"L'utilisateur {user_data.username} existe déjà.",
        )

    fake_users_db[user_data.username] = {
        "username": user_data.username,
        "full_name": user_data.full_name,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "disabled": False,
    }
    logger.info(f"Nouvel utilisateur enregistré: {user_data.username}")
    return {"message": f"Utilisateur {user_data.username} enregistré avec succès."}


@app.post("/classify", response_model=ClassificationResponse)
@timing_endpoint
async def classify_text(request: TextRequest):
    try:
        model_singleton = ModelSingleton.get_instance()
        model = model_singleton.get_model()

        probas = model.predict_proba([request.text])[0]
        prediction = model.predict([request.text])[0]

        sentiment = "positive" if prediction == 1 else "negative"
        confidence = probas[1] if prediction == 1 else probas[0]

        return ClassificationResponse(sentiment=sentiment, confidence=float(confidence))
    except Exception as e:
        logger.error(f"Erreur lors de la classification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la classification du texte",
        )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de Classification de Texte. Accédez à /docs pour la documentation."}