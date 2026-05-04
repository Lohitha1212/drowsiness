import cv2

# Read image
img = cv2.imread("input.jpg")

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image
blur = cv2.medianBlur(gray, 5)

# Detect edges
edges = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 9, 9
)

# Apply color filter
color = cv2.bilateralFilter(img, 9, 250, 250)

# Combine edges and color
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Show output
cv2.imshow("Cartoon Image", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()