from app import db

class Experiment(db.Model):
    id_ = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name_ = db.Column(db.String(64), nullable=False)
    start_date_ = db.Column(db.Date)
    type_ = db.Column(db.String(64), nullable=False)
    result_ = db.Column(db.Text)
    test_data_ = db.Column(db.Text)
    train_data_ = db.Column(db.Text)

    def __repr__(self):
        return '<Experiment %r>' % self.id_
