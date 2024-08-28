from aiohttp import content_disposition_filename
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from transformers import pipeline
from .models import Blog

@login_required
def index(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        # Handle video upload
        video_file = request.FILES['video_file']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        uploaded_file_url = fs.url(filename)
        absolute_path = fs.path(filename)

        # Extract text from video using a suitable model
        model = "facebook/wav2vec2-large-960h-lv60-self"
        pipe = pipeline(model=model)
        text = pipe(absolute_path, chunk_length_s=10)  # Processing video to text
        text_file_path = fs.path("original_text.txt")

        # Save the extracted text to a file
        with open(text_file_path, "w") as text_file:
            text_file.write(text["text"])

        # Read the extracted text for summarization
        with open(text_file_path, "r") as text_file:
            text_article = text_file.read()

        # Summarize the text using a summarization model
        summarizer = pipeline("summarization", model="google/pegasus-xsum")
        tokenizer_kwargs = {'truncation': True, 'max_length': 512}
        text_summarization = summarizer(text_article, min_length=30, do_sample=False, **tokenizer_kwargs)
        summarized_text = text_summarization[0]['summary_text']

        # Save the generated blog in the database
        blog = Blog(user=request.user, title="Generated Blog", content=summarized_text)
        blog.save()

        return redirect('saved_blogs')
    
    return render(request, 'index.html')

@login_required
def saved_blogs(request):
    # Retrieve all blogs saved by the user
    blogs = Blog.objects.filter(user=request.user)
    return render(request, 'saved_blogs.html', {'blogs': blogs})

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')
