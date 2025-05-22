#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int value;
    struct Node *next;
    struct Node *prev;
} Node;

int main() {
    srand(time(NULL)); // Seed for random number generation

    Node *head = NULL;
    Node *p, *prev = NULL; // Working pointers
    int value;
    int k;
    Node *newNode;

    // Create the first node (head)
    head = (Node *)calloc(1, sizeof(Node));
    head->value = 0;
    head->next = NULL;
    head->prev = NULL;

    int max_power = 18; // Adjust this to control max nodes (e.g., 2^10)

    for (k = 1; k <= max_power; k++) {
        int num_nodes = 1 << k; // 2^k nodes
        clock_t start = clock(); // Start time

        for (int i = 1; i < num_nodes; i++) {
            value = rand() % 1000; // Random value between 0 and 999

            // Create a new node
            newNode = (Node *)calloc(1, sizeof(Node));
            newNode->value = value;
            newNode->next = NULL;
            newNode->prev = NULL;

            // Insert in sorted order
            p = head;
            prev = NULL;

            while (p != NULL && p->value < value) {
                prev = p;
                p = p->next;
            }

            if (prev == NULL) { // Insert at head (not possible since head is always 0)
                newNode->next = head;
                head->prev = newNode;
                head = newNode;
            } else {
                newNode->next = prev->next;
                newNode->prev = prev;
                if (prev->next != NULL) {
                    prev->next->prev = newNode;
                }
                prev->next = newNode;
            }
        }

        clock_t end = clock(); // End time
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Time in seconds

        printf("List %d: Number of nodes is %d, time taken: %f seconds\n", k, num_nodes, time_taken);
    }

    // Free allocated memory
    p = head;
    while (p != NULL) {
        Node *temp = p;
        p = p->next;
        free(temp);
    }

    return 0;
}
