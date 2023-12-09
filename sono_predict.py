import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

#carrega os modelos treinados
model_sleep_duration = joblib.load('model_duration.pkl')
model_sleep_quality = joblib.load('model_quality.pkl')

#configuração base do streamlit
st.set_page_config(
    page_icon='💤',
    page_title='Qualidade do Sono',
)

#navbar
selected2 = option_menu(None, ['Início', 'Como funciona?', 'Sobre'], 
    icons=['house', 'clipboard2-data', 'info-circle'], 
    menu_icon='cast', default_index=0, orientation='horizontal')

#página inicial
if selected2 == 'Início':
    st.title('Prevendo a duração e a saúde do seu sono')

    st.markdown('''
    Preencha os dados e receba uma previsão da duração e da qualidade do seu sono com base nas informações fornecidas.
    ''')

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

    #reindexa o dataframe para que ele tenha as mesmas colunas do modelo
    user_data = user_data.reindex(columns=['Age', 'Stress Level', 'Occupation_Accountant', 'Occupation_Doctor',
       'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Manager',
       'Occupation_Nurse', 'Occupation_Sales Representative',
       'Occupation_Salesperson', 'Occupation_Scientist',
       'Occupation_Software Engineer', 'Occupation_Teacher', 'Gender_Female',
       'Gender_Male', 'Sleep Disorder_Insomnia', 'Sleep Disorder_None',
       'Sleep Disorder_Sleep Apnea'], fill_value=0)

    #faz as predições para duração e qualidade do sono
    sleep_duration_pred = model_sleep_duration.predict(user_data)
    sleep_quality_pred = model_sleep_quality.predict(user_data)

    #botão para mostrar os resultados:
    if st.button('Mostrar resultados'):
        st.write('Duração do sono: ', round(sleep_duration_pred[0], 2), ' horas.')
        st.write('Qualidade do sono: ', round(sleep_quality_pred[0], 2), ' (0 - 10).')

    st.markdown('''
    *[Liga de Data Science da Unicamp (LigaDS)](https://www.instagram.com/ligadsunicamp/)*
    ''')

#página 'Como funciona?'
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
    #### O modelo é confiável?
    Para avaliar a qualidade do modelo treinado, calculamos o **R²** e o **MSE** (Erro Quadrático Médio) para a duração e a qualidade do sono.
                ''')
    st.markdown('''
    Sim, modelo é confiável, pois apresenta um **R² de 0.85 para a duração do sono (MSE 0.10)** e **R² de 0.93 para a qualidade do sono (MSE 0.11)**.
                ''')
    

    with st.expander('Clique aqui para visualizar os gráficos', expanded=False):
        st.markdown('''
        ## Visualizando os dados
        O gráfico abaixo mostra o **nível de estresse pelo tempo dormido (em horas) juntamente com a sua qualidade (escala de 0 a 10)**:
                    ''')
        st.image('https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/imgs/estresse_tempo_qualidade.png')
        st.markdown('''
        O gráfico abaixo mostra o **tempo médio de sono por profissão** e **a qualidade média do sono por profissão**:
                    ''')
        st.image('https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/imgs/tempo_profissao.png')
        st.markdown('''
        **Tempo médio de sono levando em conta a desordem do sono**:
                    ''')
        st.image('https://raw.githubusercontent.com/thiagonarcizo/Qualidade-do-Sono/main/imgs/tempo_desordem.png')


    st.markdown('''
    ## Limitações
    O modelo, obviamente, não é perfeito, porque não leva em consideração outros fatores que podem influenciar a duração e a qualidade do sono, como a alimentação e a prática de exercícios físicos.
                ''')
    st.markdown('''
    Além disso, em algumas categorias, o dataset pode carecer de uma quantidade expressiva de informações, o que influencia na análise estatística final dos dados.
                ''')
    
#página 'Sobre'
elif selected2 == 'Sobre':
    st.title('Sobre')
    st.markdown('''
    Tratamento e análise dos dados pela **Liga de Data Science da Unicamp (LigaDS)**.
                ''')
    st.markdown('''
    Para mais informações, cheque as nossas redes sociais:
    - Instagram: [@ligadsunicamp](https://www.instagram.com/ligadsunicamp/)
    - LinkedIn: [Liga de Data Science](https://www.linkedin.com/company/liga-de-data-science/)
                ''')
    st.markdown('''
    ### Integrantes do projeto
    - Meio (suas infos aqui)
    - [Thiago Narcizo](https://www.linkedin.com/in/thiago-narcizo/)
                ''')