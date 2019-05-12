 $(document).ready(function() {

  var dropContainer = document.getElementById('drop-container');
  dropContainer.ondragover = dropContainer.ondragend = function() {
    return false;
  };

  dropContainer.ondrop = function(e) {
    e.preventDefault();
    loadAudio(e.dataTransfer.files[0])
  };

  $("#browse-button").change(function() {
    loadAudio($("#browse-button").prop("files")[0]);
  });

  $('.modal').modal({
    dismissible: false,
    ready: function(modal, trigger) {
      $.ajax({
        type: "POST",
        url:'API/',
        data: {
          'data': $('#img-card').attr('src'),
          'service': 'VC'
        },
        dataType: 'text',

        success: function(data) {
          modal.modal('close');
          loadStats(data);

        },
        error: function(data) {
            modal.modal('close');
            loadStats(data);
        }
      }).always(function() {
        modal.modal('close');
      });
    }
  });

  $('#go-back, #go-start').click(function() {
    $('#img-card').removeAttr("src");
    $('#stat-table').html('');
    switchCard(0);
  });

  $('#upload-button').click(function() {
    $('.modal').modal('open');
  });
});

switchCard = function(cardNo) {
  var containers = [".dd-container", ".uf-container", ".dt-container"];
  var visibleContainer = containers[cardNo];
  for (var i = 0; i < containers.length; i++) {
    var oz = (containers[i] === visibleContainer) ? '1' : '0';
    $(containers[i]).animate({
      opacity: oz
    }, {
      duration: 200,
      queue: false,
    }).css("z-index", oz);
  }
};

loadAudio = function(file) {
  var reader = new FileReader();
  //将文件读取到内存onload相当于ajax()里的success,文件读取成功后执行，否则执行reader.onerror()
  reader.onload = function(event) {
    $('#img-card').attr('src', event.target.result);//这里result是一个string
    $('#result').attr('src', event.target.result);
  };
  reader.readAsDataURL(file);
  switchCard(1);
};

loadStats = function(jsonData) {
  switchCard(2);
  var data = JSON.parse(jsonData);
  if (data["success"] == true) {

      var markup = `
      <span>target audio</span><br>
      <audio id="result2" controls></audio><br>
      `;
      //<audio id="result2" src="/static/audio/y_test.wav" controls></audio><br>
      $("#stat-table").append(markup);
      $('#result2').attr('src', data["result"]);

  }
};
