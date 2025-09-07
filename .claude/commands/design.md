ULTRATHINK about the design task: $ARGUMENTS

## System Design Framework

Apply these proven principles from good system design:

### Core Philosophy
"Good system design is not about clever tricks, it's about knowing how to use boring, well-tested components in the right place."

### Design Principles

1. **Minimize Stateful Components**
   - Prefer stateless services where possible
   - Centralize state management in one service
   - Stateful components can get into a bad state

2. **Database Design Excellence**
   - Create flexible but readable schemas
   - Use appropriate indexes
   - Push complex operations to the database itself
   - Distribute read queries to replicas

3. **Operation Efficiency**
   - Separate fast and slow operations
   - Use background jobs for time-consuming tasks
   - Split out the minimum amount of work needed to do something useful

4. **Strategic Caching**
   - Cache sparingly and strategically
   - Avoid caching without first attempting performance optimization
   - Use persistent storage for large-scale caching when needed

5. **Event Management**
   - Use event hubs judiciously
   - Prefer direct API calls when possible
   - Events work well for high-volume, non-time-sensitive data

6. **Observability First**
   - Log aggressively during error scenarios
   - Monitor operational metrics (p95/p99 request times)
   - Implement detailed logging for critical paths

7. **Robust Failure Handling**
   - Design robust error management
   - Implement circuit breakers
   - Create clear "fail open" or "fail closed" strategies
   - Use idempotency keys for critical operations

## For Coding Tasks

If this is a coding task, FIRST use the Task tool with subagent_type `datastructure` to:
- Define the core data structures
- Establish type annotations
- Model the domain entities and their relationships
- Ensure illegal states are unrepresentable

Then use the Task tool with subagent_type `pythonic` to:
- Define key function signatures
- Identify modern Python patterns to apply
- Plan idiomatic Python implementations
- Consider performance implications

## Detailed Design Process

ULTRATHINK through each of these questions and provide detailed answers:

### 1. Problem Understanding
- What specific problem are we solving?
- Who are the stakeholders and what are their needs?
- What are the constraints (technical, business, time)?
- What are the success criteria?

### 2. System Architecture
- What are the major components and their responsibilities?
- How do components communicate with each other?
- Where does state live and how is it managed?
- What are the data flows through the system?

### 3. Data Design
- What are the core entities and their relationships?
- How will data be stored, indexed, and accessed?
- What are the read/write patterns?
- How will data consistency be maintained?

### 4. Scalability & Performance
- What are the expected load patterns?
- Where are the potential bottlenecks?
- How will the system handle growth?
- What monitoring and alerting is needed?

### 5. Failure Modes
- What can go wrong and how likely is it?
- How will failures be detected and handled?
- What are the recovery strategies?
- How will we maintain data integrity during failures?

### 6. Implementation Strategy
- What should be built first (MVP approach)?
- How can we validate assumptions early?
- What are the key integration points?
- How will we test the system?

Remember: Focus on clarity over cleverness, and validate assumptions with evidence whenever possible.
