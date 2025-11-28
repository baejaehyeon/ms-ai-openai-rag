import streamlit as st
import os
import re # 정규표현식 모듈 추가 (필수)
from openai import AzureOpenAI
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="🎨 상황별 응답 AI 챗봇", layout="wide")
st.title("🤖 텍스트 & 이미지 생성 챗봇")

# 2. Azure OpenAI 클라이언트 설정 및 변수 정리
AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
OAI_DEPLOYMENT = "gpt-4o-mini" 

# 일반 텍스트 응답을 위한 GPT 클라이언트
client = AzureOpenAI(
    api_key=AZURE_OAI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_OAI_ENDPOINT
)

# DALL-E 이미지 생성을 위한 클라이언트 설정
# (보안을 위해 엔드포인트와 키를 환경 변수로 처리하는 것이 좋습니다.)
dalle_deployment = "dall-e-3" 
create_img_client = AzureOpenAI(
    api_version="2024-04-01-preview",
    # 하드코딩된 주소를 환경 변수로 대체 권장
    azure_endpoint="https://el22-mieta6ou-australiaeast.cognitiveservices.azure.com/",
    api_key=os.getenv("OTHER_KEY"),
)

# 3. 대화기록(Session State) 초기화 및 시스템 프롬프트 설정
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 시스템 프롬프트: AI의 이미지 생성 규칙을 명확히 정의
    st.session_state.system_prompt = {
        "role": "system",
        "content": "당신은 만능 비서입니다. 사용자가 '그려줘', '이미지', '사진', '생성' 등의 단어를 사용하여 그림 요청을 하면, 요청 내용 중 핵심 키워드만을 추출하여 **[IMAGE: 키워드]** 형식으로 응답하세요. 예시: '고양이 그려줘' -> '[IMAGE: 귀여운 고양이]', 그 외의 질문에는 일반 텍스트로 응답하세요."
    }
    # 시스템 프롬프트를 대화 목록 맨 앞에 추가 (단, 한 번만)
    st.session_state.messages.append(st.session_state.system_prompt)


# 4. 화면에 기존 대화 내용 출력
for message in st.session_state.messages:
    # 시스템 프롬프트는 화면에 출력하지 않음
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            # 저장된 내용이 이미지 URL인 경우 st.image로 출력
            if message["role"] == "assistant" and message["content"].startswith("http"):
                st.image(message["content"], caption="AI가 생성한 이미지")
            else:
                st.markdown(message["content"])

# 5. 사용자 입력 받기
if prompt := st.chat_input("무엇을 도와드릴까요?"):

    # 5-1. 사용자 메시지 화면에 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5-2. LLM 응답 생성 (이미지 생성 요청 포함)
    with st.chat_message("assistant"):
        # 모델에 전달할 전체 메시지 리스트 구성
        model_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        # 1차 호출: LLM이 텍스트 또는 [IMAGE:...] 패턴을 반환하도록 요청
        response = client.chat.completions.create(
            model=OAI_DEPLOYMENT, 
            messages=model_messages
        )
        assistant_reply = response.choices[0].message.content

        # 6. 이미지 생성 판단 및 처리 로직

        # [IMAGE: ...] 패턴을 정규표현식으로 추출 (대소문자 구분 없음)
        image_match = re.search(r"\[IMAGE:\s*(.*?)\]", assistant_reply, re.IGNORECASE)

        if image_match:
            # 이미지 생성 요청이 탐지됨
            image_prompt = image_match.group(1).strip()

            # 사용자에게 이미지 생성 중임을 알림
            with st.spinner(f"🎨 '{image_prompt}'(으)로 이미지를 생성하고 있습니다..."):

                try:
                    # DALL-E 모델 호출
                    img_response = create_img_client.images.generate(
                        model=dalle_deployment,
                        prompt=image_prompt,
                        size="1024x1024",
                        n=1,
                    )

                    # 생성된 이미지 URL 가져오기
                    image_url = img_response.data[0].url

                    # Streamlit에 이미지 표시
                    st.image(image_url, caption=f"'{image_prompt}' 이미지")

                    # 최종 응답을 이미지 URL로 저장
                    final_response = image_url

                except Exception as e:
                    error_msg = f"❌ 이미지 생성에 실패했습니다. DALL-E API 오류: {e}"
                    st.error(error_msg)
                    final_response = error_msg

        else:
            # 일반 텍스트 응답
            st.markdown(assistant_reply)
            final_response = assistant_reply

    # 7. AI 응답 저장 (이미지 URL 또는 텍스트)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
