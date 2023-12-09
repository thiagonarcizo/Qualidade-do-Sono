from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_icon='💤',
    page_title='Qualidade do Sono',
)

selected2 = option_menu(None, ['Início', 'Como funciona?', 'Sobre'], 
    icons=['house', 'clipboard2-data', 'info-circle'], 
    menu_icon='cast', default_index=0, orientation='horizontal')

if selected2 == 'Início':
    #dataset:
    uri = 'https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/data/ds.csv'
    ds = pd.read_csv(uri)

    #cópia de segurança
    df = ds.copy()

    st.title('Análise de dados da saúde do sono')

    st.markdown('''
    ## Prevendo a duração e a saúde do seu sono
    Preencha os dados e receba uma previsão da duração e da qualidade do seu sono com base nas informações fornecidas.
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

    st.markdown('''
    *Todos os créditos à [Liga de Data Science da Unicamp (LDS)](https://www.instagram.com/ligadsunicamp/)*
    ''')
elif selected2 == 'Como funciona?':
    st.title('Como funciona?')
    st.markdown('''
    ## Tratamento dos dados
    Os dados são retirados de um dataset que registra a duração e a qualidade do sono de 400 pessoas. Trabalhamos com 5 variáveis: idade, profissão, gênero, nível de estresse e distúrbio do sono.
                ''')
    st.markdown('''
    A partir dessas variáveis, treinamos um modelo de regressão linear para prever a duração e a qualidade do sono de uma pessoa com base nas informações fornecidas.
                ''')
    st.markdown('''
    O modelo é confiável, pois apresenta um **R² de 0.85 para a duração do sono (MSE 0.10)** e **R² de 0.93 para a qualidade do sono (MSE 0.11)**.
                ''')
    st.markdown('''
    ## Dados visuais
    O gráfico abaixo mostra o **nível de estresse pelo tempo dormido (em horas) juntamente com a sua qualidade (escala de 0 a 10)**:
                ''')
    st.image('https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/imgs/estresse_tempo_qualidade.png')
    st.markdown('''
    O gráfico abaixo mostra o **tempo médio de sono por profissão** e **a qualidade média do sono por profissão**:
                ''')
    st.image('https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/imgs/tempo_profissao.png')
    st.markdown('''
    ## Limitações
    O modelo, obviamente, não é perfeito, pois não leva em consideração outros fatores que podem influenciar a duração e a qualidade do sono, como a alimentação e a prática de exercícios físicos.
                ''')
    st.markdown('''
    Além disso, o dataset não apresenta muitas pessoas com idades extremas, o que pode comprometer a previsão para pessoas muito jovens ou muito velhas.
                ''')
elif selected2 == 'Sobre':
    st.title('Sobre')