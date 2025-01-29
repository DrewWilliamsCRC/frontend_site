# Frontend Site with User Auth (Flask)

This is a minimal Flask project providing:
- User registration and login (with hashed passwords via Werkzeug).
- A simple landing page (index.html) that links to your other services.

## Quickstart

1. **Python 3** is required. On Debian/Ubuntu:
   \`\`\`bash
   sudo apt-get update
   sudo apt-get install python3 python3-venv
   \`\`\`

2. **Create and activate a virtual environment**:
   \`\`\`bash
   cd frontend_site
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`

3. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Run the dev server**:
   \`\`\`bash
   python app.py
   \`\`\`
   - Access at [http://0.0.0.0:5000](http://0.0.0.0:5000).

## Production Setup (Gunicorn + Nginx)

1. Install Gunicorn:
   \`\`\`bash
   pip install gunicorn
   \`\`\`

2. Test run:
   \`\`\`bash
   gunicorn --bind 0.0.0.0:5000 wsgi:app
   \`\`\`

3. **Nginx** reverse proxy example:
   \`\`\`
   server {
       listen 80;
       server_name my-frontend.example.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   \`\`\`

4. Point your domain to this server, add SSL if desired.  

## Database Notes

- SQLite file \`users.db\` is created automatically if not present.
- Passwords are stored with Werkzeug's \`generate_password_hash\`.

## Security

- **Change** \`app.secret_key\` in \`app.py\` to a random, unique string.
- Use HTTPS in production.
- Consider stronger password rules, 2FA, etc. for real security.

