from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SelectMultipleField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import ListWidget, CheckboxInput

class InputForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField(
        'Select your gender',
        choices=[("M", "Male"),("F", "Female")],
    )
    systolic_bp = IntegerField('Systolic BP', validators=[DataRequired()])
    diastolic_bp = IntegerField('Diastolic BP', validators=[DataRequired()])
    cholesterol = SelectField(
        'Select your cholesterol level',
        choices=[
            ('Normal', 'Normal'),
            ('Borderline', 'Borderline'),
            ('High', 'High'),
            ('Very High', 'Very High')
        ],
    )
    habits = SelectMultipleField(
        'Select Your Habits',
        choices = [
            ('smokes occasionally', 'Smokes Occasionally'),
            ('smokes heavily', 'Smokes Heavily'),
            ('non-smoker', 'Non-Smoker'),
            ('smokes rarely', 'Smokes Rarely'),  
            ('smokes', 'Smokes'),
            ('drinks', 'Drinks'),
            ('drinks heavily', 'Drinks Heavily'),
            ('drinks occasionally', 'Drinks Occasionally'),
            ('no exercise', 'No Exercise'),
            ('regular exercise', 'Regular Exercise'),
            ('exercises weekly', 'Exercises Weekly'),
            ('exercises sometimes', 'Exercises Sometimes'),
            ('light exercise', 'Light Exercise'),
            ('gym twice a week', 'Gym Twice a Week'),
            ('jogs often', 'Jogs Often'),
            ('runs daily', 'Runs Daily'),
            ('swims regularly', 'Swims Regularly'),
            ('yoga weekly', 'Yoga Weekly'),
            ('cycles daily', 'Cycles Daily'),
            ('hikes weekends', 'Hikes Weekends'),
            ('runs 3x week', 'Runs 3x Week'),
            ('dances weekly', 'Dances Weekly'),
            ('walks daily', 'Walks Daily'),
            ('rock climbing', 'Rock Climbing'),
            ('soccer player', 'Soccer Player'),
            ('tennis player', 'Tennis Player'),
            ('sedentary', 'Sedentary'),
            ('no movement', 'No Movement'),
            ('bedridden', 'Bedridden'),
            ('wheelchair-bound', 'Wheelchair-Bound'),
            ('immobile', 'Immobile'),
            ('stressed', 'Stressed'),
            ('desk job', 'Desk Job'),
            ('office job', 'Office Job'),
            ('light gardening', 'Light Gardening'),
            ('walks dog', 'Walks Dog'),
            ('golf occasionally', 'Golf Occasionally'),
            ('active lifestyle', 'Active Lifestyle'),
            ('athlete', 'Athlete'),
            ('martial arts', 'Martial Arts'),
            ('pilates', 'Pilates'),
            ('yoga enthusiast', 'Yoga Enthusiast'),
            ('fitness class', 'Fitness Class'),
            ('gym member', 'Gym Member'),
            ('gym regular', 'Gym Regular')
        ],
        option_widget=CheckboxInput(),   
        widget=ListWidget(prefix_label=False),
    )
    submit = SubmitField('Predict')
