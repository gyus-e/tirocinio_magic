from flask import Flask
import routes
from utils import DB
from environ import DB_URL

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
DB.init_app(app)

with app.app_context():
    DB.create_all()

app.register_blueprint(routes.configurations_blueprint)
app.register_blueprint(routes.cag_blueprint, url_prefix="/configurations")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)