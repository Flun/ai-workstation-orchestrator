설치 방법

1. chrome://extensions 를 엽니다.
2. 우측 상단의 개발자 모드를 켭니다.
3. "압축해제된 확장 프로그램을 로드합니다"를 누릅니다.
4. /home/flux/main_server_new/chrome_instagram_cookie_bridge 폴더를 선택합니다.
   (확장 프로그램 내용을 수정한 경우 chrome://extensions 에서 해당 확장의 [새로고침] 버튼을 누릅니다)
5. 확장 아이콘을 누릅니다.
   - "서버 주소" 칸에 Studio 서버의 IP(포트)를 입력합니다. 기본값: 192.168.1.243
     (IP만 입력하면 :8999 포트를 붙여 http://<IP>:8999 로 요청합니다. 입력값은 자동 저장됩니다.)
   - "Instagram 쿠키 동기화": Instagram 로그인 쿠키를 서버로 보냅니다 (Dataset Studio 수집용).
   - "X/Twitter 쿠키 동기화": x.com 로그인 쿠키를 서버로 보냅니다 (Media Studio 다운로드용).
   Chrome에서 해당 사이트에 로그인한 상태여야 합니다.

참고
- 기본 서버(127.0.0.1 / 192.168.1.243) 외의 다른 주소를 사용할 경우,
  동기화 버튼을 누를 때 Chrome가 해당 서버에 대한 권한 허용을 한 번 물어봅니다. [허용]을 누르면 이후부터는 묻지 않습니다.

이 확장은 instagram.com, x.com, twitter.com 쿠키만 Studio 서버 공용 저장소로 전송합니다.
동기화된 X 쿠키는 Media Studio의 X/Twitter 미디어 다운로드에서 함께 사용됩니다.
