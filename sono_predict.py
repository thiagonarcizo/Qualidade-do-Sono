import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

#traduzindo do português para o inglês cada entrada de string (o modelo foi treinado em inglês)
def translate_prof(prof):
    if prof == 'Engenheiro de Software' or prof == 'Engenheira de Software':
        prof = 'Software Engineer'
    elif prof == 'Médico' or prof == 'Médica':
        prof = 'Doctor'
    elif prof == 'Representante de Vendas' or prof == 'Representante de Vendas':
        prof = 'Sales Representative'
    elif prof == 'Professor' or prof == 'Professora':
        prof = 'Teacher'
    elif prof == 'Enfermeiro' or prof == 'Enfermeira':
        prof = 'Nurse'
    elif prof == 'Engenheiro' or prof == 'Engenheira':
        prof = 'Engineer'
    elif prof == 'Contador' or prof == 'Contadora':
        prof = 'Accountant'
    elif prof == 'Cientista' or prof == 'Cientista':
        prof = 'Scientist'
    elif prof == 'Advogado' or prof == 'Advogada':
        prof = 'Lawyer'
    elif prof == 'Vendedor' or prof == 'Vendedora':
        prof = 'Salesperson'
    else:
        prof = 'Manager'
    return prof

def translate_sleep_disorder(sleep_disorder):
    if sleep_disorder == 'Apneia do sono':
        sleep_disorder = 'Sleep Apnea'
    elif sleep_disorder == 'Insônia':
        sleep_disorder = 'Insomnia'
    else:
        sleep_disorder = 'None'
    return sleep_disorder

def translate_gender(genero):
    if genero == 'Masculino':
        genero = 'Male'
    else:
        genero = 'Female'
    return genero

#carrega os modelos treinados
model_sleep_duration = joblib.load('model_duration.pkl')
model_sleep_quality = joblib.load('model_quality.pkl')

#configuração base do streamlit
st.set_page_config(
    page_icon='💤',
    page_title='Qualidade do Sono',
    layout='wide'
)

left_co, last_co = st.columns(2)
#logo
with last_co:
    st.image('imgs/dslogo.png', width=100)

#navbar
with left_co:
    selected2 = option_menu(None, ['Início', 'Como funciona?', 'Sobre'], 
        icons=['house', 'clipboard2-data', 'info-circle'],
        menu_icon='cast', default_index=0, orientation='horizontal')

#página inicial
if selected2 == 'Início':
    st.title('Prevendo a duração e a saúde do seu sono')

    st.markdown('''
    Preencha os dados e receba uma previsão da duração e da qualidade do seu sono com base nas informações fornecidas.
    ''')

    st.divider()

    #faz as predições para duração do sono com base nos inputs pessoais
    col1, col2 = st.columns(2)

    masc_prof = ['Engenheiro de Software', 'Médico', 'Representante de Vendas', 'Professor', 'Enfermeiro', 'Engenheiro', 'Contador', 'Cientista', 'Advogado', 'Vendedor', 'Gerente']
    fem_prof = ['Engenheira de Software', 'Médica', 'Representante de Vendas', 'Professora', 'Enfermeira', 'Engenheira', 'Contadora', 'Cientista', 'Advogada', 'Vendedora', 'Gerente']

    with col1:
        idade = st.number_input(r'''$\textsf{\Large Sua idade:}$''', min_value=0, max_value=100, value=0, step=1)
        genero = st.selectbox(r'''$\textsf{\Large Seu gênero:}$''', ('Masculino', 'Feminino'))
        prof = st.selectbox(r'''$\textsf{\Large Sua profissão:}$''', masc_prof) if genero == 'Masculino' else st.selectbox('Insira a sua profissão', fem_prof)
    with col2:
        stress_level = st.number_input(r'''$\textsf{\Large Seu nível de estresse (0-10):}$''', min_value=0, max_value=10, value=0, step=1)
        sleep_disorder = st.selectbox(r'''$\textsf{\Large Distúrbio do sono:}$''', ('Apneia do sono', 'Insônia', 'Nenhum'), index=2)
    
    #cria um dataframe com os dados do usuário
    user = pd.DataFrame({'Age': [idade], 'Occupation': [translate_prof(prof)], 'Gender': [translate_gender(genero)], 'Stress Level': [stress_level], 'Sleep Disorder': [translate_sleep_disorder(sleep_disorder)]})

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

    sleep_duration_vs_avg = sleep_duration_pred[0] - 7.132085561497325
    sleep_quality_vs_avg = sleep_quality_pred[0] - 7.31283422459893

    #botão para mostrar os resultados:
    if st.button(r'''$\textsf{\LARGE Mostrar Resultados}$'''):
        if genero == 'Masculino':
            st.latex(r'\textsf{\Large Tempo de sono: '+'\Huge '+str(round(sleep_duration_pred[0], 2))+'\Large h}')
            st.latex(r'\textsf{\Large Qualidade do sono: '+'\Huge '+str(round(sleep_quality_pred[0], 2))+'\Large (0-10)}')
        else:
            st.latex(r'\textsf{\Large Tempo de sono: '+'\Huge '+str(round(sleep_duration_pred[0], 2))+'\Large h}')
            st.latex(r'\textsf{\Large Qualidade do sono: '+'\Huge '+str(round(sleep_quality_pred[0], 2))+'\Large (0-10)}')
        st.latex(r'\textsf{\Large Você tem '+'\Large '+str(round(abs(sleep_duration_vs_avg), 2))+' horas de sono a mais do que a média}' if sleep_duration_vs_avg > 0 else (r'\textsf{\Large Você tem '+'\Large '+str(round(abs(sleep_duration_vs_avg), 2))+' horas de sono a menos do que a média}'))
        st.latex(r'\textsf{\Large Você pontuou '+'\Large '+str(round(abs(sleep_quality_vs_avg), 2))+' a mais do que a média}' if sleep_quality_vs_avg > 0 else (r'\textsf{\Large Você pontuou '+'\Large '+str(round(abs(sleep_quality_vs_avg), 2))+' a menos do que a média}'))

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
    A média geral de tempo de sono é: 7.13 horas e a pontuação média geral de qualidade do sono é: 7.31.
                ''')
    st.divider()
    st.markdown('''
    #### O modelo é confiável?
    Para avaliar a qualidade do modelo treinado, calculamos o **R²** e o **MSE** (Erro Quadrático Médio) para a duração e a qualidade do sono.
                ''')
    st.markdown('''
    Sim, o modelo é confiável, pois apresenta um **R² de 0.85 para a duração do sono (MSE 0.10)** e **R² de 0.93 para a qualidade do sono (MSE 0.11)**.
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
    st.divider()
    st.title('Integrantes do projeto')
    st.markdown('''
    - Meio (suas infos aqui)
    - [Thiago Narcizo](https://www.linkedin.com/in/thiago-narcizo/)
                ''')