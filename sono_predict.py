from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st

#ds:
uri = 'data/ds.csv'
ds = pd.read_csv(uri)

#cópia de segurança
df = ds.copy()

st.title('Análise de dados da saúde do sono')

st.markdown('''
## Prevendo a duração e a saúde do seu sono
Preencha os dados corretamente e receba uma previsão da duração e da qualidade do seu sono.
''')

#prepara os dados para o modelo
X = df[['Age', 'Occupation', 'Gender', 'Stress Level', 'Sleep Disorder']]
y_sleep_duration = df['Sleep Duration']
y_sleep_quality = df['Quality of Sleep']

#converte as variáveis categóricas em numéricas
X = pd.get_dummies(X)

#separa os dados em treino e teste
X_train, X_test, y_train_sleep_duration, y_test_sleep_duration, y_train_sleep_quality, y_test_sleep_quality = train_test_split(X, y_sleep_duration, y_sleep_quality, test_size=0.2, random_state=42)

#treina o modelo para duração do sono
model_sleep_duration = LinearRegression()
model_sleep_duration.fit(X_train, y_train_sleep_duration)

#treina o modelo para qualidade do sono
model_sleep_quality = LinearRegression()
model_sleep_quality.fit(X_train, y_train_sleep_quality)

#faz as predições para duração do sono com base nos inputs pessoais
idade = st.number_input('Insira a sua idade:', min_value=0, max_value=100, value=0, step=1)
genero = st.selectbox('Selecione seu gênero:', ('Male', 'Female'))
prof = st.selectbox('Insira a sua profissão', ('Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'))
stress_level = st.number_input('Seu nível de estresse (0 - 10):', min_value=0, max_value=10, value=0, step=1)
sleep_disorder = st.selectbox('Em qual dessas condições de sono você melhor se enquadra?', ('None', 'Sleep Apnea', 'Insomnia'))

#cria um dataframe com os dados do usuário
user = pd.DataFrame({'Age': [idade], 'Occupation': [prof], 'Gender': [genero], 'Stress Level': [stress_level], 'Sleep Disorder': [sleep_disorder]})

#converte as variáveis categóricas em numéricas (do usuário)
user_data = pd.get_dummies(user)

#reindexa o dataframe para que ele tenha as mesmas colunas do X_train
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

#faz as predições para duração e qualidade do sono
sleep_duration_pred = model_sleep_duration.predict(user_data)
sleep_quality_pred = model_sleep_quality.predict(user_data)

#calcula o R^2 and MSE para duração e qualidade do sono
sleep_duration_r2 = r2_score(y_test_sleep_duration, model_sleep_duration.predict(X_test))
sleep_duration_mse = mean_squared_error(y_test_sleep_duration, model_sleep_duration.predict(X_test))
sleep_quality_r2 = r2_score(y_test_sleep_quality, model_sleep_quality.predict(X_test))
sleep_quality_mse = mean_squared_error(y_test_sleep_quality, model_sleep_quality.predict(X_test))

#botão para mostrar os resultados:
if st.button('Mostrar resultados'):
    st.write('Duração do sono: ', round(sleep_duration_pred[0], 2), ' horas.')
    st.write('Qualidade do sono: ', round(sleep_quality_pred[0], 2), ' (0 - 10).')
    #st.write('Duração do sono: ', sleep_duration_prediction[0], ' horas')
    #st.write('Qualidade do sono: ', sleep_quality_prediction[0], ' (0 - 10)')
