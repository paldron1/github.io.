# Paldron LLC Touch Free HMI .

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redact Text from Video</title>
</head>
<body>
    <h1>Redact Words from Video Feed</h1>

    <form id="redactForm">
        <label for="wordsToRedact">Enter words to redact (comma-separated):</label>
        <input type="text" id="wordsToRedact" name="wordsToRedact" placeholder="word1, word2, word3">
        <button type="submit">Submit</button>
    </form>

    <div id="response"></div>

    <script>
        document.getElementById('redactForm').addEventListener('submit', function(e) {
            e.preventDefault();

            const words = document.getElementById('wordsToRedact').value.split(',').map(word => word.trim());
            
            fetch('/redact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ words_to_redact: words })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('response').innerText = data.message;
            })
            .catch(error => {
                document.getElementById('response').innerText = 'Error: ' + error.message;
            });
        });
    </script>
</body>
</html>
