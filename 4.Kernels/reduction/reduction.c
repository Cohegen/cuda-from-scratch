#include <stdio.h>

int main(void)
{
    float sum = 0.0f;
    float inputs[5] = {7.0f, 8.0f, 10.02f, 0.56f, 0.5f};

    for (int i = 0; i < 5; ++i) {
        sum += inputs[i];
    }

    printf("Sum: %.2f\n", sum);

    return 0;
}
