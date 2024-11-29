from app import create_app
import os
app = create_app()

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(debug=True)
