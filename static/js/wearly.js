



        $("#next_page").on("click",function(){
          if($("#name_input>input").val() != '' && $("#age_input>input").val() != ''){
              $("#first").css('display','none');
              $("#second").css('display','block');
          }else{
              alert("별명과 나이를 적어주세요!")
          }
      });

      document.addEventListener('keydown', function(event) {
        if (event.keyCode === 13) {
            event.preventDefault();
        };
      }, true);


