import re

with open('app.py', 'r') as f:
    content = f.read()

# Replace the inner loop logic in classify_colors
old_loop = '''        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = center_x + (j - 1) * SPACING
                y = center_y + (i - 1) * SPACING
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                
                # Crop a 32x32 square around the center point (x, y)
                half_size = 16
                y1 = max(0, y - half_size)
                y2 = min(height, y + half_size)
                x1 = max(0, x - half_size)
                x2 = min(width, x + half_size)
                
                crop = img_rgb[y1:y2, x1:x2]
                if crop.shape[0] != 32 or crop.shape[1] != 32:
                    crop = cv2.resize(crop, (32, 32))
                
                pil_img = Image.fromarray(crop)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    _, predicted = torch.max(outputs, 1)
                    pred_idx = predicted.item()
                    color = COLOR_MAP.get(pred_idx, "W")
                
                colors.append(color)
                positions.append({'x': int(x), 'y': int(y)})'''

new_loop = '''        tensors = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = center_x + (j - 1) * SPACING
                y = center_y + (i - 1) * SPACING
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                
                # Crop a 32x32 square around the center point (x, y)
                half_size = 16
                y1 = max(0, y - half_size)
                y2 = min(height, y + half_size)
                x1 = max(0, x - half_size)
                x2 = min(width, x + half_size)
                
                crop = img_rgb[y1:y2, x1:x2]
                if crop.shape[0] != 32 or crop.shape[1] != 32:
                    crop = cv2.resize(crop, (32, 32))
                
                pil_img = Image.fromarray(crop)
                tensors.append(transform(pil_img))
                positions.append({'x': int(x), 'y': int(y)})
        
        # Perform batched tensor inference (shape: [9, 3, 32, 32])
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predicted = torch.max(outputs, 1)
        
        for pred_idx in predicted:
            colors.append(COLOR_MAP.get(pred_idx.item(), "W"))'''

content = content.replace(old_loop, new_loop)

with open('app.py', 'w') as f:
    f.write(content)
