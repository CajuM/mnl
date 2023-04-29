master: [376000] running\_loss=0.019801878702593968; closs=0.023282647132873535; audible=0.00992882251739502; success=0.998828113079071;
decoder-l2: [2596000] running\_loss=0.013381955973192817; closs=0.009219411760568619; audible=0.009885132312774658; success=0.9990234375;
with just one linear layer in the decoder, it failes to converge, two linear layers with a RelU in-between seems to work best
