import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 한글 글꼴
plt.rc('font',family='malgun gothic')

def basic():
    # 파일 불러오기
    df = pd.read_csv('한국전력공사_지역별 전기차 현황정보_20230331.csv', encoding='euc-kr')

    # 피벗 해제 (열의 데이터 변환)
    df_melt = pd.melt(df,id_vars='기준일', value_vars=['서울', '인천', '경기', '강원', '충북', '충남', '대전', '세종', '경북', '대구', '전북','전남', '광주', '경남', '부산', '울산', '제주', '합계'],var_name='지역',value_name='자동차수')  

    # 년, 월 파생변수 생성
    df_melt['년'] = df_melt['기준일'].str[:4]
    df_melt['월'] = df_melt['기준일'].str[5:7]

    return df_melt

def region_mean(df_melt):
    # *************************************************
    # 지역별(행) 연도별(2022,2023) 자동차수 평균 계산 - pivot_table
    year_region_da = round(pd.pivot_table(df_melt, index='년', columns='지역', values='자동차수',aggfunc='mean'), 1)
    st.dataframe(year_region_da)

    year_region_da = year_region_da.T  # 행,열 전환
    region_query = year_region_da[year_region_da.index != '합계']
   
    ax = region_query.plot(kind='bar', rot=0)
    fig = ax.get_figure()
    st.pyplot(fig)

def mean_2023(df_melt):
# *************************************************
    # 2023년 1-3월 자동차수 분석
    df_melt_2023 = df_melt[(df_melt['년']=='2023') & (df_melt['지역'] != '합계')]
    df_2023 = pd.pivot_table(df_melt_2023,index='지역', columns='월',values='자동차수',aggfunc='mean')
    st.dataframe(df_2023.T) #행,열 전환 돌려서 프린트

    ax = df_2023.plot(kind='bar',rot=0)
    fig = ax.get_figure()
    st.pyplot(fig)

def quarter_mean(df_melt):
    # *************************************************
    # 2022년 분기별 자동차수 분석
    df_2022 = df_melt[df_melt['년']=='2022']

    df_2022['월'] = df_2022['월'].astype(int)

    # numpy 안에서 조건비교함수. 파생변수 생성
    df_2022['분기'] = np.where((df_2022['월']>=1) & (df_2022['월'] <=3),'1분기',
                            np.where((df_2022['월']>=4) & (df_2022['월'] <=6),'2분기',
                            np.where((df_2022['월']>=7) & (df_2022['월'] <=9),'3분기','4분기')))

    # 분기별, 지역별 통계 (분기-열)
    df_2022_da = round(pd.pivot_table(df_2022,index='지역', columns='분기',values='자동차수',aggfunc='mean'),0)
    # 합계는 제외
    df_2022_da= df_2022_da[df_2022_da.index != '합계']
    st.dataframe(df_2022_da.T)

    ax = df_2022_da.plot(kind='bar',rot=0)
    fig = ax.get_figure()
    st.pyplot(fig)

    # **************************************************
    # group-by
    #df_2022_da2 = df_2022.groupby(['지역','분기'])[['자동차수']].mean().reset_index()
    #df_2022_da2 = df_2022_da2[df_2022_da2['지역'] != '합계']
    #print(df_2022_da2)


# main 실행
def elec_exe():
    menu = st.selectbox('분석내용',['선택','지역별/연도별 분석','2023년 지역별 분석','2022년 분기별 분석'])
    return_df_melt = basic()

    if menu == '지역별/연도별 분석':
        region_mean(return_df_melt)
    elif menu == '2023년 지역별 분석':
        mean_2023(return_df_melt)
    elif menu == '2022년 분기별 분석':
        quarter_mean(return_df_melt)
    else:
        st.image('모지스 웨딩.jpg',width=300)

if __name__=='__main__':
    elec_exe()


# while True:
#     menu = int(input('메뉴입력 (1: 지역별,연도별 분석, 2: 2023년 분석, 3: 2022년 분기별 분석, 0번: 종료)'))
#     if menu == 1:
#         region_mean(return_df_melt)
#     elif menu == 2:
#         mean_2023(return_df_melt)
#     elif menu == 3:  
#         quarter_mean(return_df_melt)
#     elif menu == 0:  
#         break
#     else:
#         print('입력 오류')


