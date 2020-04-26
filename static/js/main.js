$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

  $('#imageUpload').click(function () {
    $('#workspace').hide();
    $('#response').hide();
  });


    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $("#workspace").css("display", "block");
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
              console.log(data);
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
              //  $('#result').text(' Result:  ' + data);
                var results = JSON.parse(JSON.stringify(data));
                results['Probability Score'] = Number(results['Probability Score']).toFixed(2);
                $("#response tbody tr").empty();
                $('#response').find('tbody').append("<tr><td>"+results['Class']+"</td><td>"+results['Probability Score']+"</td><td>"+results['ROP']+"</td></tr>");
              $('#response').show();

            },
        });
    });

});
