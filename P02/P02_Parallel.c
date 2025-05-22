#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <omp.h>


typedef struct Node {
    int value;
    struct Node *next;
    struct Node *prev;
    omp_lock_t lock;  
} Node;

int main(int argc, char *argv[]) {
    
 
    
      int  num_threads = atoi(argv[1]);
      int  num_inserts =  atoi(argv[2]);
      num_inserts = 1 << num_inserts;
      //printf("Number of Threads: %d" , num_threads);
      //printf("Number of nodes: %d", num_inserts);
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Create a global dummy head node with a sentinel value
    Node *head = malloc(sizeof(Node));
    if (!head) {
        perror("Failed to allocate head node");
        exit(EXIT_FAILURE);
    }
    head->value = 0;  // Sentinel value
    head->next = NULL;
    head->prev = NULL;
    omp_init_lock(&head->lock);  // Initialize the lock for the head node

    // Start timing the parallel insertion phase
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // Each thread uses its own seed for thread-safe random number generation
        unsigned int seed = (unsigned int) time(NULL) ^ omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_inserts; i++) {
            int value = rand_r(&seed) % 1000;  // Random value between 0 and 999

            // Allocate and initialize the new node
            Node *newNode = malloc(sizeof(Node));
            if (!newNode) {
                fprintf(stderr, "Memory allocation failed in thread %d\n", omp_get_thread_num());
                continue;
            }
            newNode->value = value;
            newNode->next = NULL;
            newNode->prev = NULL;

            // Traverse the list with hand-over-hand (lock coupling) locking
            Node *prev = head;
            omp_set_lock(&prev->lock);
            Node *curr = prev->next;
            if (curr != NULL) {
                omp_set_lock(&curr->lock);
            }

            // Walk the list until the proper insertion spot is found
            while (curr != NULL && curr->value < value) {
                omp_unset_lock(&prev->lock);
                prev = curr;
                curr = curr->next;
                if (curr != NULL) {
                    omp_set_lock(&curr->lock);
                }
            }

            // Insert newNode between 'prev' and 'curr'
            newNode->prev = prev;
            newNode->next = curr;
            prev->next = newNode;
            if (curr != NULL) {
                curr->prev = newNode;
            }

            // Initialize the new node's lock for future operations
            omp_init_lock(&newNode->lock);

            // Release the locks
            if (curr != NULL) {
                omp_unset_lock(&curr->lock);
            }
            omp_unset_lock(&prev->lock);
        }
    }  // End of parallel region

    // End timing and calculate the time taken
    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    // Print the timing result
    printf("Threads: %d, List Size: %d, Time taken for parallel insertion: %e seconds\n", 
           num_threads, num_inserts, time_taken);

    // Free the list (destroying locks along the way)
    Node *curr = head;
    while (curr != NULL) {
        Node *temp = curr;
        curr = curr->next;
        omp_destroy_lock(&temp->lock);
        free(temp);
    }

    return 0;
}
